"""Run auditable examples of all four GPPAF algorithm modules on a CPU.

Process data here are explicitly algebraic example fixtures; author-supplied
data are exercised separately by scripts/train_process.py. No reported paper
number is used as a target, hard-coded output, or acceptance threshold.
"""
from pathlib import Path
import sys,time,json,argparse,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from shapely.geometry import Polygon,LineString
from gppaf import __version__
from gppaf.process import TCDSFNet,WINDOWS,calibrate,design
from gppaf.data import bootstrap_signs,layer_trends
from gppaf.geometry import build_graph,path_feasible,coverage
from gppaf.objectives import Objective
from gppaf.hgts import PAHGTS
from gppaf.search import beam_search,reinforce_step
from gppaf.smoothing import smooth_path,SmoothConfig,derivatives
from gppaf.layers import within_layer,connect_layers,LayerConfig
from gppaf.io import save_json,save_path_csv,path_svg,environment


def process_example(out,epochs):
    torch.manual_seed(123);rng=np.random.RandomState(123)
    model=TCDSFNet(hidden=24,context=12)
    batches={};arrays={};signs={}
    for d in WINDOWS:
        lo,hi=map(np.array,WINDOWS[d]);r=rng.uniform(.1,.9,size=(64,len(lo)))
        x=lo+(hi-lo)*r
        # Example forward relation; dimensions are illustrative mm.
        y=np.column_stack([7+2*r[:,0]-r[:,1]+r[:,3],2+.5*r[:,0]-.3*r[:,1]-.2*r[:,3]])
        if d=='multi': y+=np.column_stack([-.3*r[:,4],.1*r[:,4]])
        arrays[d]=(x,y,r)
        batches[d]=tuple(torch.tensor(a,dtype=torch.float32) for a in (x[:40],y[:40],np.ones(40)))
        signs[d]=bootstrap_signs(r[:40],y[:40],repeats=1000,seed=123)
    layer_y=arrays['multi'][1][:,None,:]+np.arange(-4.5,5)[:,None]*np.array([.035,-.012])[None,:]
    layer_y=torch.tensor(layer_y,dtype=torch.float32)
    model.fit_scalers({d:b[1] for d,b in batches.items()},layer_y[:40]-layer_y[:40].mean(1,keepdim=True))
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    trace=[]
    for epoch in range(epochs):
        loss,terms=model.losses(batches,signs,(batches['multi'][0],layer_y[:40]),layer_signs=layer_trends(layer_y[:40].numpy()))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        trace.append(float(loss.detach()))
    model.eval();cal={};report={}
    for d,(x,y,_) in arrays.items():
        tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
        cal[d]=calibrate(model,tx[40:52],ty[40:52],d,64,123)
        result=design(model,ty[52:],d,cal[d],64,321)
        prediction=model.predict_geometry(model.normalize(result['selected'],d),d)
        report[d]=dict(target_geometry=ty[52:],selected=result['selected'],predicted_geometry=prediction,
                       finite_candidates=bool(torch.isfinite(result['candidates']).all()),
                       best_of_m_normalized_mae=float(((result['candidates']-tx[52:,None,:])/getattr(model,d+'_range')).abs().mean(-1).min(-1)[0].mean()))
    _,delta=model.predict_layers(model.normalize(batches['multi'][0],'multi'))
    report.update(loss_initial=trace[0],loss_final=trace[-1],zero_mean_deviation_max=float(delta.mean(1).abs().max().detach()),
                  scope='Algebraic example fixture for exercising the TC-DSFNet implementation',
                  implementation_version=__version__)
    torch.save(dict(config=model.config,state_dict=model.state_dict(),scope=report['scope'],implementation_version=__version__),str(out/'process_demo.pt'))
    save_json(out/'process.json',report);save_json(out/'process_calibration.json',cal);save_json(out/'process_loss.json',trace)
    return dict(loss_initial=trace[0],loss_final=trace[-1],layer_zero_mean_max=report['zero_mean_deviation_max'])


def path_example(out):
    data=json.loads((Path(__file__).parent/'hole_path.json').read_text(encoding='utf-8'))
    graph=build_graph(**{k:v for k,v in data.items() if k not in ['provenance','start','end']})
    model=PAHGTS(hidden=24,heads=2,layers=2);baseline=copy.deepcopy(model).eval()
    for p in baseline.parameters(): p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    obj=Objective(scales=(10,1,10,10,10));training=[]
    for _ in range(3): training.append(reinforce_step(graph,model,baseline,obj,optimizer,0,3))
    result=beam_search(graph,model,obj,0,3,width=16)
    if result.path is None: raise RuntimeError('Example hole path failed')
    assert path_feasible(graph,result.path,0,3)
    save_path_csv(out/'path.csv',graph.points[result.path]);path_svg(out/'path.svg',graph.region,graph.points[result.path],'PA-HGTS: computed hole-avoiding path')
    torch.save(dict(config=model.config,state_dict=model.state_dict(),training_scope='Small example graph, 3 training updates'),str(out/'hgts_demo.pt'))
    report=dict(status=result.status,node_order=result.path,points=graph.points,objective=result.objective,
                metrics=obj.metrics(graph,result.path),coverage_fraction=coverage(graph,result.path),training=training)
    save_json(out/'path.json',report);return dict(status=result.status,nodes=len(result.path),objective=result.objective)


def smoothing_example(out):
    safe=Polygon([(-4,-2),(4,-2),(4,5),(-4,5)])
    points=np.array([[-3.,0],[0,0],[0,3],[3,3]])
    result=smooth_path(points,safe,SmoothConfig(deviation=.5,max_curvature=20))
    assert result['status']=='smoothed'
    save_path_csv(out/'smooth.csv',result['points']);path_svg(out/'smooth.svg',safe,result['points'],'PA-LCCS: computed local quintic smoothing')
    save_json(out/'smoothing.json',result)
    return dict(status=result['status'],smoothed_corners=len(result['curves']))


def layers_example(out):
    safe=Polygon([(-2,-2),(12,-2),(12,8),(-2,8)])
    # One open infill component plus two separate closed contour components.
    components=[np.array([[0.,1],[2,1]]),np.array([[4.,0],[6,0],[6,2],[4,2],[4,0]]),
                np.array([[8.,4],[10,4],[10,6],[8,6],[8,4]])]
    cfg=LayerConfig(cut_count=3,retained_states=12,max_connection=14,max_interlayer=14,max_tilt_from_vertical=89)
    states=[within_layer(components,safe,z=z,config=cfg) for z in [0.,1.,2.]]
    result=connect_layers(states,safe,cfg)
    if result['status']!='feasible': raise RuntimeError('Example multilayer connection failed')
    save_path_csv(out/'multilayer.csv',result['points']);save_json(out/'layers.json',result)
    save_json(out/'layer_states.json',[[dict(cost=s.cost,component_order=[q.component for q in s.traversals],
              cuts=[q.cut for q in s.traversals],directions=[q.direction for q in s.traversals],points=s.points) for s in layer] for layer in states])
    path_svg(out/'layer0.svg',safe,states[0][result['state_indices'][0]].points,'HELS-DP: computed component connections')
    return dict(status=result['status'],retained_states=[len(s) for s in states],cost=result['cost'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('runs/examples'))
    parser.add_argument('--epochs',type=int,default=60);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(123)
    start=time.perf_counter();summary={}
    for name,fn in [('process',lambda:process_example(args.output,args.epochs)),('path',lambda:path_example(args.output)),
                    ('smoothing',lambda:smoothing_example(args.output)),('layers',lambda:layers_example(args.output))]:
        t=time.perf_counter();summary[name]=fn();summary[name]['seconds']=time.perf_counter()-t
        print(name,summary[name],flush=True)
    summary['environment']=environment();summary['total_seconds']=time.perf_counter()-start
    summary['scope']='Executable examples for the four GPPAF algorithm modules'
    summary['implementation_version']=__version__
    save_json(args.output/'summary.json',summary)
