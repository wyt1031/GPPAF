"""Geometry -> process candidates -> feasible path -> smoothing -> layer DP.

The exported XYZ path is a research interchange format. Controller kinematics,
robot collision checking and executable robot programs belong to the separate
application integration and are not provided by this research-code repository.
"""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gppaf.process import TCDSFNet,design
from gppaf.hgts import PAHGTS
from gppaf.geometry import build_graph
from gppaf.objectives import Objective
from gppaf.search import beam_search
from gppaf.smoothing import smooth_path,SmoothConfig
from gppaf.layers import within_layer,connect_layers,LayerConfig
from gppaf.io import save_json,save_path_csv,path_svg

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--process-run',type=Path,required=True)
    p.add_argument('--path-checkpoint',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--width',type=float,required=True);p.add_argument('--height',type=float,required=True)
    p.add_argument('--layers',type=int,default=3);p.add_argument('--beam',type=int,default=32)
    p.add_argument('--output',type=Path,default=Path('runs/pipeline'));args=p.parse_args()
    if min(args.width,args.height,args.layers)<=0: raise ValueError('Positive dimensions and layer count required')
    torch.set_num_threads(1);start=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(str(args.process_run/'model.pt'),map_location='cpu')
    proc=TCDSFNet(**checkpoint['config']);proc.load_state_dict(checkpoint['state_dict']);proc.eval()
    cal=json.loads((args.process_run/'calibration.json').read_text(encoding='utf-8'))
    target=torch.tensor([[args.width,args.height]],dtype=torch.float32)
    process={d:design(proc,target,d,cal[d])['selected'] for d in ['single','multi']}
    source=json.loads(args.geometry.read_text(encoding='utf-8'))
    settings={k:v for k,v in source.items() if k in ['outer','holes','fill_rate','phase','k_neighbors','max_scale','points']}
    # Explicit co-design coupling: target bead width determines path clearance
    # and sampling interval, and target height determines layer elevation.
    graph=build_graph(bead_width=args.width,**settings)
    ckpt=torch.load(str(args.path_checkpoint),map_location='cpu');net=PAHGTS(**ckpt['config']);net.load_state_dict(ckpt['state_dict']);net.eval()
    objective=Objective(scales=(len(graph.points)*graph.spacing,1,len(graph.points),len(graph.points),len(graph.points)))
    route=beam_search(graph,net,objective,source.get('start',0),source.get('end',len(graph.points)-1),args.beam)
    report=dict(process=process,path_status=route.status,path_reason=route.reason)
    if route.path is not None:
        smoothed=smooth_path(graph.points[route.path],graph.safe,SmoothConfig(bead_width=args.width,deviation=.3*args.width,max_curvature=8/args.width))
        components=[smoothed['points']]+[np.asarray(c,float) for c in source.get('additional_components',[])]
        cfg=LayerConfig(distance_scale=graph.spacing,interlayer_scale=args.height,max_connection=4*graph.spacing,max_interlayer=4*graph.spacing)
        states=[within_layer(components,graph.safe,z=i*args.height,config=cfg) for i in range(args.layers)]
        linked=connect_layers(states,graph.safe,cfg)
        report.update(smoothing_status=smoothed['status'],smoothing_corners=smoothed['corners'],layer_status=linked['status'])
        if linked['status']=='feasible':
            save_path_csv(args.output/'path_xyz.csv',linked['points']);save_json(args.output/'layer_connection.json',linked)
        path_svg(args.output/'layer_xy.svg',graph.region,smoothed['points'],'GPPAF computed layer centerline')
    report['elapsed_seconds']=time.perf_counter()-start
    save_json(args.output/'pipeline.json',report);print(str(args.output))
