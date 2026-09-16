"""Train, calibrate and evaluate TC-DSFNet on the packaged GPPAF process data.

Validation selection, preprocessing and augmentation are isolated from test
groups. The default formal evaluation uses 100 epochs and M=256 candidates.
"""
from pathlib import Path
import argparse,csv,sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gppaf.process import TCDSFNet,calibrate,design
from gppaf.data import grouped_split,augment_training,bootstrap_signs,layer_trends
from gppaf.io import save_json,environment,fingerprint


def load_mean(path,p):
    with path.open(encoding='utf-8') as f: rows=list(csv.reader(f))
    return (np.array([r[2:2+p] for r in rows[1:]],dtype=float),np.array([r[2+p:4+p] for r in rows[1:]],dtype=float),
            np.array([r[1] for r in rows[1:]]),[r[0] for r in rows[1:]])


def load_layers(path):
    with path.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    grouped={}
    for r in rows: grouped.setdefault(r['condition_id'],[]).append(r)
    x=[];y=[];groups=[]
    for key,records in sorted(grouped.items()):
        records.sort(key=lambda r:int(r['layer']))
        if [int(r['layer']) for r in records]!=list(range(1,11)): raise ValueError('Each condition must have exactly layers 1..10')
        params=['current_A','speed_cm_min','frequency_Hz','amplitude_mm','cooling_s']
        settings=np.array([[float(r[p]) for p in params] for r in records])
        if not np.allclose(settings,settings[:1]): raise ValueError('Conditions vary inside a layer group')
        x.append(settings[0]);y.append([[float(r['width_mm']),float(r['height_mm'])] for r in records]);groups.append(records[0]['group_id'])
    return np.array(x),np.array(y),np.array(groups)


def run(args):
    start=time.perf_counter();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    measured={d:load_mean(args.data/(d+'.csv'),p) for d,p in [('single',4),('multi',5)]}
    layers=load_layers(args.data/'layers.csv') if (args.data/'layers.csv').exists() else None
    all_groups=np.concatenate([v[2] for v in measured.values()]+([layers[2]] if layers is not None else []))
    # Split a joint group universe once: shared physical settings cannot cross folds.
    unique=np.unique(all_groups); partition=grouped_split(unique,args.fold,args.folds,args.seed)
    group_sets={k:set(unique[v]) for k,v in partition.items()}
    splits={d:{k:np.array([i for i,g in enumerate(v[2]) if g in gs],dtype=int) for k,gs in group_sets.items()} for d,v in measured.items()}
    if any(len(ix)==0 for split in splits.values() for ix in split.values()): raise ValueError('A domain split is empty; choose another fold or more data')
    model=TCDSFNet(hidden=args.hidden,context=args.context)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    scaler_y={d:tensor(v[1][splits[d]['train']]) for d,v in measured.items()}
    layer_batch=None;layer_sign=(False,False);dev=None
    layer_split={}
    if layers is not None:
        layer_split={k:[i for i,g in enumerate(layers[2]) if g in gs] for k,gs in group_sets.items()}
        ix=layer_split['train']; lx,ly=layers[0][ix],layers[1][ix]
        layer_batch=(tensor(lx),tensor(ly));dev=tensor(ly-ly.mean(1,keepdims=True))
        layer_sign=layer_trends(ly,repeats=args.bootstrap,seed=args.seed)
    model.fit_scalers(scaler_y,dev)
    training={};signs={};parents={}
    for d,(x,y,groups,ids) in measured.items():
        tr=splits[d]['train']
        ax,ay,w,parent=augment_training(x[tr],y[tr],seed=args.seed)
        training[d]=(tensor(ax),tensor(ay),tensor(w));parents[d]=[[ids[tr[i]],ids[tr[j]]] for i,j in parent]
        signs[d]=bootstrap_signs(x[tr],y[tr],groups[tr],repeats=args.bootstrap,seed=args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    trace=[];best=float('inf');best_epoch=0
    for epoch in range(args.epochs):
        model.train();loss,terms=model.losses(training,signs,layer_batch=layer_batch,layer_signs=layer_sign,
                   weights=(10.,2.,0. if args.ablation=='no_trend' else .5,0. if args.ablation=='no_layer' else .5),
                   deviation_weight=0. if args.ablation=='no_layer' else 1.)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        model.eval()
        with torch.no_grad():
            validation=0.
            for d,(x,y,_,_) in measured.items():
                ix=splits[d]['validation'];vx,vy=tensor(x[ix]),tensor(y[ix])
                geo=((model.predict_geometry(model.normalize(vx,d),d)-vy)/getattr(model,d+'_yscale')).abs().mean()
                validation+=float(-model.log_prob(vx,vy,d).mean()+10*geo)
        row=dict(epoch=epoch+1,total=float(loss.detach()),validation=validation,**{k:float(v.detach()) for k,v in terms.items()});trace.append(row)
        if validation<best:
            best=validation;best_epoch=epoch+1
            torch.save(dict(config=model.config,state_dict=model.state_dict(),epoch=best_epoch,training_protocol='GPPAF grouped training protocol'),str(out/'model.pt'))
        if (epoch+1)%max(1,args.epochs//5)==0: print('epoch %d/%d loss %.4f val %.4f'%(epoch+1,args.epochs,float(loss.detach()),validation),flush=True)
    checkpoint=torch.load(str(out/'model.pt'),map_location='cpu');model.load_state_dict(checkpoint['state_dict']);model.eval()
    calibrations={};evaluation={}
    for d,(x,y,groups,ids) in measured.items():
        val=splits[d]['validation'];test=splits[d]['test']
        cal=calibrate(model,tensor(x[val]),tensor(y[val]),d,args.candidates,args.seed+10)
        # Layer smoothness is unusable if layer data were not available.
        if d=='multi' and (layers is None or args.ablation=='no_layer'):
            cal['weights'][3]=0.;total=sum(cal['weights']);cal['weights']=[v/total for v in cal['weights']]
            cal['layer_scoring']='disabled because layer supervision is not active for this run'
        calibrations[d]=cal
        result=design(model,tensor(y[test]),d,cal,args.candidates,args.seed+20)
        span=getattr(model,d+'_range')
        errors=((result['candidates']-tensor(x[test])[:,None,:])/span).abs().mean(-1)
        selected=((result['selected']-tensor(x[test]))/span).abs().mean(-1)
        lo,hi=torch.quantile(result['candidates'],.05,dim=1),torch.quantile(result['candidates'],.95,dim=1)
        coverage=((tensor(x[test])>=lo)&(tensor(x[test])<=hi)).float().mean().item()
        evaluation[d]=dict(test_samples=len(test),best_of_m_mae_percent=100*errors.min(-1)[0].mean().item(),
                 selected_mae_percent=100*selected.mean().item(),marginal_90_coverage=coverage,
                 absolute_coverage_error=abs(coverage-.9),per_sample_best_percent=100*errors.min(-1)[0],
                 test_ids=[ids[i] for i in test],selected=result['selected'])
    save_json(out/'calibration.json',calibrations)
    save_json(out/'evaluation.json',dict(evaluation=evaluation,
             elapsed_seconds=time.perf_counter()-start,best_epoch=best_epoch,layer_supervision=layers is not None,ablation=args.ablation,environment=environment()))
    save_json(out/'splits.json',dict(groups={k:sorted(v) for k,v in group_sets.items()},
              rows={d:{k:[measured[d][3][i] for i in ix] for k,ix in part.items()} for d,part in splits.items()},layer_rows=layer_split))
    save_json(out/'augmentation_parents.json',parents);save_json(out/'training_history.json',trace)
    save_json(out/'config.json',vars(args));save_json(out/'trend_signs.json',dict(parameter=signs,layer=layer_sign))
    print(json.dumps(dict(output=str(out),best_epoch=best_epoch,layer_supervision=layers is not None,seconds=time.perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed'));p.add_argument('--output',type=Path,default=Path('runs/process'))
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--fold',type=int,default=0);p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--hidden',type=int,default=64);p.add_argument('--context',type=int,default=32)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--candidates',type=int,default=256);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--ablation',choices=['full','no_trend','no_layer'],default='full')
    run(p.parse_args())
