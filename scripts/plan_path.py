"""Plan a polygon-with-holes JSON instance, with an optional PA-HGTS checkpoint."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gppaf.geometry import build_graph,coverage
from gppaf.objectives import Objective
from gppaf.hgts import PAHGTS
from gppaf.search import beam_search
from gppaf.io import save_json,save_path_csv,path_svg,environment


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,default=Path('runs/path'))
    p.add_argument('--beam',type=int,default=32)
    p.add_argument('--lambda-cost',type=float,default=1.0,help='Eq. (35) cumulative-objective weight')
    p.add_argument('--lambda-bound',type=float,default=0.1,help='Eq. (35) lower-bound weight')
    p.add_argument('--local-cost-weight',type=float,default=1.,help='Eq. (36) gamma multiplying c_loc')
    p.add_argument('--local-rounds',type=int,default=2)
    p.add_argument('--local-top-edges',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(1);start=time.perf_counter();source=json.loads(args.input.read_text(encoding='utf-8'))
    settings={k:source[k] for k in ['outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale','points'] if k in source}
    graph=build_graph(**settings);model=None
    if args.checkpoint:
        checkpoint=torch.load(str(args.checkpoint),map_location='cpu')
        model=PAHGTS(**checkpoint['config']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    objective=Objective(scales=(len(graph.points)*graph.spacing,1,len(graph.points),len(graph.points),len(graph.points)))
    result=beam_search(graph,model,objective,source.get('start',0),source.get('end',len(graph.points)-1),args.beam,lambda_cost=args.lambda_cost,lambda_bound=args.lambda_bound,local_cost_weight=args.local_cost_weight,local_rounds=args.local_rounds,local_top_edges=args.local_top_edges)
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status=result.status,reason=result.reason,path=result.path,objective=result.objective,
                model='PA-HGTS checkpoint' if model else 'Explicit process-cost baseline (no neural checkpoint)',
                nodes=len(graph.points),edges=len(graph.edge_index),expanded=result.expanded,
                elapsed_seconds=time.perf_counter()-start,environment=environment())
    if result.path is not None:
        report['metrics']=objective.metrics(graph,result.path);report['coverage_fraction']=coverage(graph,result.path)
        save_path_csv(args.output/'path.csv',graph.points[result.path]);path_svg(args.output/'path.svg',graph.region,graph.points[result.path])
    save_json(args.output/'result.json',report);print(json.dumps(dict(status=result.status,output=str(args.output))))
