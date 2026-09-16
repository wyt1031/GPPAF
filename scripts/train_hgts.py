"""Train PA-HGTS using explicit train/validation datasets or the bundled regression curriculum.

The formal benchmark set is separate from the small regression curriculum.
"""
from pathlib import Path
import sys,argparse,copy,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gppaf.geometry import build_graph
from gppaf.hgts import PAHGTS
from gppaf.objectives import Objective
from gppaf.search import reinforce_step,rollout
from gppaf.io import save_json,environment
from gppaf.path_data import load_instances,require_disjoint


def curriculum():
    cases=[]
    for nx,ny in [(2,3),(3,2),(3,3),(4,2)]:
        pts=[(x,y) for y in range(ny) for x in range(nx)]
        g=build_graph([(-.5,-.5),(nx-.5,-.5),(nx-.5,ny-.5),(-.5,ny-.5)],
                      bead_width=.4,points=pts,max_scale=4,k_neighbors=8)
        cases.append((g,0,len(pts)-1))
    hole=build_graph([(-.5,-.5),(2.5,-.5),(2.5,2.5),(-.5,2.5)],
                    holes=[[(.8,.8),(1.2,.8),(1.2,1.2),(.8,1.2)]],bead_width=.4,
                    points=[(x,y) for y in range(3) for x in range(3) if (x,y)!=(1,1)],max_scale=4,k_neighbors=8)
    cases.append((hole,0,3));return cases


def case_objective(graph):
    return Objective(scales=(len(graph.points)*graph.spacing,1,len(graph.points),len(graph.points),len(graph.points)))


@torch.no_grad()
def evaluate(model,cases,failure_penalty):
    """Deterministic greedy validation; retain every failure in the denominator."""
    was_training=model.training;model.eval();rows=[]
    try:
        for case_id,(graph,start,end) in enumerate(cases):
            objective=case_objective(graph)
            path,_,_=rollout(graph,model,objective,start,end,False)
            rows.append(dict(case=case_id,feasible=path is not None,path=path,
                             objective=objective(graph,path) if path is not None else failure_penalty))
    finally:
        model.train(was_training)
    failures=sum(not row['feasible'] for row in rows)
    return dict(cases=rows,failures=failures,total=len(rows),
                success_rate=1.-failures/len(rows),
                mean_penalized_objective=float(np.mean([row['objective'] for row in rows])))


def run(args):
    if (args.instances is None)!=(args.validation_instances is None):
        raise ValueError('Supply both --instances and --validation-instances, or neither for the regression curriculum')
    if args.epochs<1 or args.hidden<4 or args.hidden%2 or not np.isfinite(args.lr) or args.lr<=0:
        raise ValueError('Require epochs >= 1, an even hidden width >= 4, and a positive finite learning rate')
    if not np.isfinite(args.failure_penalty) or args.failure_penalty<=0:
        raise ValueError('Validation failure penalty must be positive and finite')
    validation_cases=[];data_manifest=None
    if args.instances is not None:
        cases,train_manifest,train_records=load_instances(args.instances)
        validation_cases,val_manifest,val_records=load_instances(args.validation_instances)
        require_disjoint(train_manifest,val_manifest)
        data_manifest=dict(training=train_manifest,validation=val_manifest,
            separation='Exact polygon groups and supplied group_id values are disjoint; no test data are read')
    else:
        cases=curriculum()
    start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=PAHGTS(hidden=args.hidden,heads=2,layers=2);baseline=copy.deepcopy(model).eval()
    for p in baseline.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);history=[];validation_history=[]
    best_state=None;best_key=None;best_epoch=None
    def validate(epoch):
        nonlocal best_state,best_key,best_epoch
        record=evaluate(model,validation_cases,args.failure_penalty)
        validation_history.append(dict(epoch=epoch,**record))
        # First minimize failures, then cost; retain the earlier epoch on ties.
        key=(record['failures'],record['mean_penalized_objective'])
        if best_key is None or key<best_key:
            best_key=key;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
    if validation_cases: validate(0)
    for epoch in range(args.epochs):
        for case_id,(graph,s,e) in enumerate(cases):
            obj=case_objective(graph)
            row=reinforce_step(graph,model,baseline,obj,optimizer,s,e,regret_weight=getattr(args,'regret_weight',.1),positive_regret_weight=getattr(args,'positive_regret_weight',1.))
            history.append(dict(epoch=epoch+1,case=case_id,**row))
        # A frozen model snapshot is an independent baseline for the next block.
        if (epoch+1)%5==0:
            baseline.load_state_dict(model.state_dict());print('epoch %d, updates %d'%(epoch+1,sum(r['updated'] for r in history)),flush=True)
        if validation_cases: validate(epoch+1)
    args.output.mkdir(parents=True,exist_ok=True)
    training_scope=('supplied train/validation JSON instances' if data_manifest else 'bundled regression curriculum')
    selection=('Validation: lowest failure count, then lowest mean penalized objective; earliest epoch on ties' if validation_cases else
               'Last epoch of bundled regression curriculum; no validation-based model selection')
    if validation_cases:
        torch.save(dict(config=model.config,state_dict=model.state_dict(),epoch=args.epochs,training_scope=training_scope),str(args.output/'last_model.pt'))
        model.load_state_dict(best_state)
        (args.output/'inputs').mkdir(parents=True,exist_ok=True)
        (args.output/'inputs/train.json').write_bytes(args.instances.read_bytes())
        (args.output/'inputs/validation.json').write_bytes(args.validation_instances.read_bytes())
        save_json(args.output/'dataset_manifest.json',data_manifest)
        save_json(args.output/'validation_history.json',validation_history)
    torch.save(dict(config=model.config,state_dict=model.state_dict(),training_scope=training_scope,
                    selected_epoch=best_epoch if validation_cases else args.epochs,
                    selection=selection,dataset_manifest=data_manifest),str(args.output/'model.pt'))
    save_json(args.output/'training_history.json',history)
    save_json(args.output/'run.json',dict(config=vars(args),environment=environment(),elapsed_seconds=time.perf_counter()-start,
                updates=sum(r['updated'] for r in history),attempts=len(history),
                sampled_failures=sum(r.get('sampled_failed',False) for r in history),
                baseline_failures=sum(r.get('baseline_failed',False) for r in history),
                selected_epoch=best_epoch if validation_cases else args.epochs,selection=selection,
                train_cases=len(cases),validation_cases=len(validation_cases),
                training_scope=training_scope,regret_weight=getattr(args,'regret_weight',.1),positive_regret_weight=getattr(args,'positive_regret_weight',1.),
                evaluation_scope='Validation is used for model selection, not an independent test estimate' if validation_cases else
                                 'Bundled regression curriculum for training-path execution checks'))
    print(str(args.output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--hidden',type=int,default=32);p.add_argument('--seed',type=int,default=2026);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--instances',type=Path,help='JSON list of training instances; paired validation file is required')
    p.add_argument('--validation-instances',type=Path,help='Disjoint JSON validation instances, never used for gradient updates')
    p.add_argument('--failure-penalty',type=float,default=1000.,help='Predeclared value recorded for every failed validation rollout')
    p.add_argument('--regret-weight',type=float,default=.1,help='Overall coefficient of the regret-head loss')
    p.add_argument('--positive-regret-weight',type=float,default=1.,help='Eq. (37) positive-class weight omega_plus')
    p.add_argument('--output',type=Path,default=Path('runs/hgts'));run(p.parse_args())
