"""Run RSM/SVR/RF/cVAE/Affine-cINN and TC-DSFNet on saved grouped splits.

All methods use the same measured test rows and candidate count. Classical
models share the same bounded multistart inverse search.
"""
from pathlib import Path
import sys,argparse,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gppaf.baselines import classical_process,classical_inverse,ConditionalVAE,AffineCINN
from gppaf.process import TCDSFNet,WINDOWS,design
from gppaf.data import augment_training
from gppaf.io import save_json,environment
from train_process import load_mean


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/processed'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('runs/process_baselines'))
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--candidates',type=int,default=256)
    p.add_argument('--test-limit',type=int,default=0);p.add_argument('--seed',type=int,default=2026);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    split=json.loads((args.run/'splits.json').read_text());ckpt=torch.load(str(args.run/'model.pt'),map_location='cpu')
    proposed=TCDSFNet(**ckpt['config']);proposed.load_state_dict(ckpt['state_dict']);proposed.eval()
    cal=json.loads((args.run/'calibration.json').read_text());records=[]
    for domain,dim in [('single',4),('multi',5)]:
        x,y,groups,ids=load_mean(args.data/(domain+'.csv'),dim)
        train=[i for i,k in enumerate(ids) if k in set(split['rows'][domain]['train'])]
        test=[i for i,k in enumerate(ids) if k in set(split['rows'][domain]['test'])]
        if args.test_limit: test=test[:args.test_limit]
        ax,ay,weights,_=augment_training(x[train],y[train],seed=args.seed)
        lo,hi=map(np.array,WINDOWS[domain]);span=hi-lo;ym=y[train].mean(0);ys=np.maximum(y[train].std(0),1e-6)
        for method in ['RSM','SVR','RF','cVAE','Affine-cINN','TC-DSFNet']:
            start=time.perf_counter()
            if method in ['RSM','SVR','RF']:
                # Classical fit uses measured rows to avoid treating interpolation
                # as new independent experimental information.
                model=classical_process(method,x[train],y[train],args.seed)
                candidates=np.array([classical_inverse(model,target,lo,hi,ys,args.candidates,args.seed+i) for i,target in enumerate(y[test])])
            elif method in ['cVAE','Affine-cINN']:
                model=ConditionalVAE(dim,hidden=32) if method=='cVAE' else AffineCINN(dim,hidden=32)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                rho=torch.tensor((x[train]-lo)/span,dtype=torch.float32);condition=torch.tensor((y[train]-ym)/ys,dtype=torch.float32)
                for epoch in range(args.epochs):
                    loss=model.loss(rho,condition);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                model.eval()
                with torch.no_grad(): candidates=lo+span*model.sample(torch.tensor((y[test]-ym)/ys,dtype=torch.float32),args.candidates,torch.Generator().manual_seed(args.seed)).numpy()
            else:
                candidates=design(proposed,torch.tensor(y[test],dtype=torch.float32),domain,cal[domain],args.candidates,args.seed)['candidates'].numpy()
            errors=np.abs((candidates-x[test,None,:])/span).mean(-1).min(-1)
            qlo,qhi=np.quantile(candidates,.05,axis=1),np.quantile(candidates,.95,axis=1)
            coverage=float(((x[test]>=qlo)&(x[test]<=qhi)).mean())
            record=dict(domain=domain,method=method,best_of_m_mae_percent=float(100*errors.mean()),per_sample_percent=100*errors,
                         coverage_90=coverage,coverage_interpretation='latent sampling' if method not in ['RSM','SVR','RF'] else 'multistart optimizer spread, not probabilistic calibration',
                         elapsed_seconds=time.perf_counter()-start,test_ids=[ids[i] for i in test],candidates=args.candidates)
            records.append(record);print(domain,method,record['best_of_m_mae_percent'],flush=True)
    save_json(args.output/'results.json',dict(records=records,config=vars(args),environment=environment(),
        protocol='All methods use identical grouped test rows and the configured candidate count; runtime is reported per method.'))
