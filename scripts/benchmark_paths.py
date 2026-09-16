"""Evaluate constrained baselines and PA-HGTS on the GPPAF path benchmark.

Every failed run remains in the table and contributes the fixed predeclared
feasibility penalty.
"""
from pathlib import Path
import argparse,json,sys,time,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gppaf.geometry import build_graph,path_feasible,coverage
from gppaf.objectives import Objective
from gppaf.hgts import PAHGTS
from gppaf.search import beam_search,local_search
from gppaf.baselines import nearest_neighbor,constrained_aco
from gppaf.evaluation import paired_comparison
from gppaf.io import save_json,environment


def example_suite():
    cases=[]
    for nx,ny,holes_at in [(3,3,[]),(3,3,[(1,1)]),(4,3,[]),(4,3,[(1,1)]),
                          (5,3,[(1,1),(3,1)]),(4,4,[(1,1)]),(5,4,[(1,1),(3,2)]),(5,4,[])]:
        holes=[[(x-.2,y-.2),(x+.2,y-.2),(x+.2,y+.2),(x-.2,y+.2)] for x,y in holes_at]
        points=[(x,y) for y in range(ny) for x in range(nx) if (x,y) not in holes_at]
        case=dict(outer=[(-.5,-.5),(nx-.5,-.5),(nx-.5,ny-.5),(-.5,ny-.5)],holes=holes,points=points,
                  bead_width=.4,k_neighbors=8,max_scale=4,start=0,end=(3 if nx==ny==3 and holes else len(points)-1))
        cases.append(case)
    return cases


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instances',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('runs/path_benchmark'))
    p.add_argument('--beam',type=int,default=16);p.add_argument('--failure-penalty',type=float,default=1000);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--lambda-cost',type=float,default=1.0,help='Eq. (35) cumulative-objective weight')
    p.add_argument('--lambda-bound',type=float,default=0.1,help='Eq. (35) remaining-cost lower-bound weight')
    p.add_argument('--local-cost-weight',type=float,default=1.,help='Eq. (36) gamma multiplying c_loc')
    p.add_argument('--local-rounds',type=int,default=2,help='Feasible local-search passes')
    p.add_argument('--local-top-edges',type=int,default=8,help='Top-priority path edges considered per local-search pass')
    p.add_argument('--aco-ants',type=int,default=6)
    p.add_argument('--aco-iterations',type=int,default=5)
    p.add_argument('--aco-evaporation',type=float,default=0.2)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    cases=json.loads(args.instances.read_text(encoding='utf-8')) if args.instances else example_suite()
    ckpt=torch.load(str(args.checkpoint),map_location='cpu');model=PAHGTS(**ckpt['config']);model.load_state_dict(ckpt['state_dict']);model.eval()
    rows=[];path_records=[]
    for index,case in enumerate(cases):
        graph=build_graph(**{k:v for k,v in case.items() if k in ['outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale','points']})
        s,e=case.get('start',0),case.get('end',len(graph.points)-1)
        scale=(len(graph.points)*graph.spacing,1,len(graph.points),len(graph.points),len(graph.points))
        objective=Objective(scales=scale);length_only=Objective(weights=(1,0,0,0,0),scales=scale)
        def rgls():
            result=beam_search(graph,None,objective,s,e,args.beam,lambda_cost=args.lambda_cost,lambda_bound=args.lambda_bound,refine=False)
            if result.path is not None:
                with torch.no_grad(): regret=model.encode(graph)[3].cpu().numpy()
                result.path=local_search(graph,result.path,objective,regret=regret,rounds=args.local_rounds,top_edges=args.local_top_edges,local_cost_weight=args.local_cost_weight)
                result.objective=objective(graph,result.path)
            return result
        methods={'NN':lambda:nearest_neighbor(graph,objective,s,e),
                 'ACO':lambda:constrained_aco(graph,objective,s,e,ants=args.aco_ants,iterations=args.aco_iterations,seed=args.seed+index,evaporation=args.aco_evaporation),
                 'G-beam':lambda:beam_search(graph,None,length_only,s,e,args.beam,lambda_cost=args.lambda_cost,lambda_bound=args.lambda_bound,refine=False),
                 'P-beam':lambda:beam_search(graph,None,objective,s,e,args.beam,lambda_cost=args.lambda_cost,lambda_bound=args.lambda_bound,refine=False),
                 'R-GLS':rgls,
                 'PA-HGTS':lambda:beam_search(graph,model,objective,s,e,args.beam,lambda_cost=args.lambda_cost,lambda_bound=args.lambda_bound,local_cost_weight=args.local_cost_weight,local_rounds=args.local_rounds,local_top_edges=args.local_top_edges)}
        for name,call in methods.items():
            start=time.perf_counter();result=call();seconds=time.perf_counter()-start
            valid=result.path is not None and path_feasible(graph,result.path,s,e)
            value=objective(graph,result.path) if valid else args.failure_penalty
            metrics=objective.metrics(graph,result.path) if valid else [None]*5
            rows.append(dict(instance=index,instance_id=case.get('id',str(index)),group_id=case.get('group_id',''),
                method=name,nodes=len(graph.points),holes=len(case.get('holes',[])),start=s,end=e,
                fill_rate=case.get('fill_rate',''),phase_x=(case.get('phase') or ['', ''])[0],phase_y=(case.get('phase') or ['', ''])[1],
                feasible=valid,objective=value,path_length=metrics[0],turn_metric=metrics[1],
                boundary_deviation=metrics[2],spacing_deviation=metrics[3],heat_accumulation=metrics[4],
                seconds=seconds,coverage=coverage(graph,result.path) if valid else 0))
            path_records.append(dict(instance=index,instance_id=case.get('id',str(index)),method=name,path=result.path,status=result.status))
        print('instance %d/%d, nodes %d'%(index+1,len(cases),len(graph.points)),flush=True)
    names=list(methods);values={name:np.array([r['objective'] for r in rows if r['method']==name]) for name in names}
    reference=values.pop('PA-HGTS');stats=paired_comparison(reference,values,seed=args.seed)
    summary={}
    for name in names:
        selected=[r for r in rows if r['method']==name]; successful=[r for r in selected if r['feasible']]
        def mean_success(key):
            return float(np.mean([r[key] for r in successful])) if successful else None
        summary[name]=dict(total_instances=len(selected),feasible_instances=len(successful),
            success_rate=float(np.mean([r['feasible'] for r in selected])),
            mean_penalized_objective=float(np.mean([r['objective'] for r in selected])),
            mean_path_length=mean_success('path_length'),mean_turn_metric=mean_success('turn_metric'),
            mean_boundary_deviation=mean_success('boundary_deviation'),mean_spacing_deviation=mean_success('spacing_deviation'),
            mean_heat_accumulation=mean_success('heat_accumulation'),mean_coverage=mean_success('coverage'),
            total_seconds=float(sum(r['seconds'] for r in selected)))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'per_instance.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(args.output/'paths.json',path_records);save_json(args.output/'instances.json',cases)
    save_json(args.output/'summary.json',dict(summary=summary,paired_statistics=stats,config=vars(args),environment=environment(),
        benchmark='user-supplied benchmark' if args.instances else 'bundled executable examples',
        protocol='Common geometric feasibility and final objective. Beam width shared; timings and all failures are reported.'))
