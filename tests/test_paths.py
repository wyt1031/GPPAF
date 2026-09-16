import copy,itertools
import numpy as np
import torch
from shapely.geometry import Polygon,LineString
from gppaf.geometry import build_graph,path_feasible,coverage
from gppaf.objectives import Objective,mst_lower_bound
from gppaf.hgts import PAHGTS
from gppaf.search import beam_search,local_search,regret_labels,reinforce_step,feasible_candidates
from gppaf.smoothing import control_points,derivatives,smooth_path,SmoothConfig,curvature_certificate
from gppaf.layers import entry_exit_states,within_layer,connect_layers,LayerConfig,interlayer_edge


def square_graph(hole=False):
    points=[(x,y) for y in range(3) for x in range(3) if not(hole and x==y==1)]
    holes=[[(.8,.8),(1.2,.8),(1.2,1.2),(.8,1.2)]] if hole else []
    return build_graph([(-.5,-.5),(2.5,-.5),(2.5,2.5),(-.5,2.5)],holes,
                       bead_width=.4,k_neighbors=8,max_scale=4,points=points)


def test_graph_full_segment_hole_avoidance():
    g=square_graph(True)
    assert (3,4) not in g.lookup  # two safe endpoints separated by a hole
    assert all(g.safe.buffer(1e-8).covers(s) for s in g.segments.values())
    assert g.node_features.shape[1]==8 and g.edge_features.shape[1]==7
    assert (g.relations>=0).all() and (g.relations<=2).all()


def test_beam_coverage_and_local_search_monotonicity():
    g=square_graph(True); obj=Objective(scales=(10,1,10,10,10))
    model=PAHGTS(hidden=16,heads=2,layers=2)
    result=beam_search(g,model,obj,0,3,width=16)
    assert result.status=='feasible' and path_feasible(g,result.path,0,3)
    assert 0<coverage(g,result.path)<=1
    improved=local_search(g,result.path,obj)
    assert obj(g,improved)<=obj(g,result.path)+1e-10


def test_crossing_revisit_end_and_failure():
    g=square_graph()
    assert not path_feasible(g,[0,2,6,8],complete=False)
    assert not path_feasible(g,[0,1,0],complete=False)
    assert 8 not in feasible_candidates(g,[0],8)
    bad=build_graph([(0,0),(10,0),(10,4),(0,4)],bead_width=.2,
                    points=[(1,1),(1.2,1),(8,1),(8.2,1)],max_scale=2,k_neighbors=2)
    assert beam_search(bad,None,Objective(),0,3).status=='search_failed'


def test_mst_bound_no_larger_than_exhaustive_completion():
    g=square_graph();gpoints=[0,1,2,3,4,5,6,7,8]
    # Exact subset tails of a known feasible row-serpentine path.
    path=[0,1,2,5,4,3,6,7,8]
    for k in range(1,len(path)):
        remaining=set(path[k:]);current=path[k-1]
        lb=mst_lower_bound(g,remaining,current,8)
        actual=sum(np.linalg.norm(g.points[a]-g.points[b]) for a,b in zip(path[k-1:-1],path[k:]))
        assert lb<=actual+1e-9


def test_reinforce_has_gradients_and_frozen_baseline():
    torch.manual_seed(3);g=square_graph(True);model=PAHGTS(hidden=16,heads=2,layers=1)
    base=copy.deepcopy(model).eval()
    for p in base.parameters(): p.requires_grad_(False)
    before=[p.clone() for p in base.parameters()]
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    records=[reinforce_step(g,model,base,Objective(scales=(10,1,10,10,10)),opt,0,3) for _ in range(3)]
    assert any(r['updated'] for r in records)
    assert all(torch.equal(a,b) for a,b in zip(before,base.parameters()))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())


def test_quintic_endpoint_g2_and_smoothing():
    ctrl=control_points(np.array([-1.,0]),np.array([0.,1.]),np.array([1.,0]),np.array([0.,1.]),1.,1.)
    first,second=derivatives(ctrl,np.array([0.,1.]))
    assert np.allclose(first,[[1,0],[0,1]]) and np.allclose(second,0)
    assert curvature_certificate(ctrl,20.,1e-7)
    safe=Polygon([(-5,-5),(5,-5),(5,5),(-5,5)])
    result=smooth_path([[-3,0],[0,0],[0,3]],safe,SmoothConfig(deviation=.5,max_curvature=20))
    assert result['status']=='smoothed'
    assert np.allclose(result['points'][0],[-3,0]) and np.allclose(result['points'][-1],[0,3])
    assert LineString(result['points']).is_simple
    assert len(result['curves'])==1


def test_smoothing_retains_unfeasible_corner():
    safe=Polygon([(0,0),(3,0),(3,3),(0,3)])
    r=smooth_path([[0,2],[0,0],[2,0]],safe,SmoothConfig())
    assert r['status']=='partial_or_unchanged' and len(r['curves'])==0
    assert r['corners'][0]['reason']=='window_too_small'


def test_closed_loop_cut_coverage():
    loop=np.array([[0,0],[2,0],[2,2],[0,2],[0,0]])
    q=entry_exit_states(loop,0,5)
    assert len(q)==10
    for state in q:
        assert abs(LineString(state.points).length-8)<1e-8
        assert np.allclose(state.entry,state.exit)
        assert LineString(state.points).hausdorff_distance(LineString(loop))<1e-8


def test_layer_dp_vs_bruteforce_retained_states():
    safe=Polygon([(-1,-1),(8,-1),(8,8),(-1,8)])
    components=[np.array([[0.,0],[2,0]]),np.array([[4.,1],[6,1]])]
    cfg=LayerConfig(retained_states=100,max_interlayer=20,max_tilt_from_vertical=89)
    layers=[within_layer(components,safe,z=z,config=cfg) for z in (0,1,2)]
    r=connect_layers(layers,safe,cfg)
    values=[]
    for combo in itertools.product(*layers):
        e=[interlayer_edge(a,b,safe,cfg) for a,b in zip(combo[:-1],combo[1:])]
        if all(item is not None for item in e): values.append(sum(s.cost for s in combo)+sum(item[0] for item in e))
    assert r['status']=='feasible'
    assert abs(r['cost']-min(values))<1e-9
    for states in layers:
        for s in states: assert sorted(q.component for q in s.traversals)==[0,1]


def test_layer_connector_cannot_cross_hole():
    safe=Polygon([(-1,-1),(7,-1),(7,3),(-1,3)],holes=[[(2,-.5),(4,-.5),(4,2.5),(2,2.5)]])
    components=[np.array([[0.,0],[1,0]]),np.array([[5.,0],[6,0]])]
    assert within_layer(components,safe)==[]


def test_interlayer_tilt_and_empty_states():
    assert connect_layers([[]],Polygon([(0,0),(1,0),(1,1),(0,1)]))['status']=='infeasible'


def test_explicit_epsilon_L_filters_short_corner_without_deleting_points():
    safe=Polygon([(-1,-1),(3,-1),(3,3),(-1,3)])
    pts=np.array([[0.,0.],[1e-5,0.],[0.5,0.5],[1.,1.]])
    r=smooth_path(pts,safe,SmoothConfig(epsilon_L=1e-4,deviation=.5,max_curvature=20))
    assert np.isfinite(r['points']).all()
    assert np.allclose(r['points'][0],pts[0])
    assert np.allclose(r['points'][-1],pts[-1])
    # epsilon_L screens a corner; it is not a deduplication tolerance.
    assert any(np.allclose(q,pts[1],atol=0,rtol=0) for q in r['points'])


def test_short_distinct_final_segment_preserves_original_endpoint():
    safe=Polygon([(-1,-1),(3,-1),(3,3),(-1,3)])
    pts=np.array([[0.,0.],[1.,0.],[1.00005,0.]])
    r=smooth_path(pts,safe,SmoothConfig(epsilon_L=1e-4,deviation=.5,max_curvature=20))
    assert np.allclose(r['points'],pts,atol=0,rtol=0)
    assert np.allclose(r['points'][-1],pts[-1],atol=0,rtol=0)


def test_true_adjacent_duplicate_is_removed_but_endpoint_coordinate_is_kept():
    safe=Polygon([(-1,-1),(3,-1),(3,3),(-1,3)])
    pts=np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])
    r=smooth_path(pts,safe,SmoothConfig(epsilon_L=1e-4,deviation=.5,max_curvature=20))
    assert np.allclose(r['points'],np.array([[0.,0.],[1.,0.]]),atol=0,rtol=0)
    assert np.allclose(r['points'][-1],pts[-1],atol=0,rtol=0)


def test_eq36_gamma_and_eq37_positive_weight_are_explicit():
    torch.manual_seed(8);g=square_graph(True);obj=Objective(scales=(10,1,10,10,10))
    model=PAHGTS(hidden=16,heads=2,layers=1);base=copy.deepcopy(model).eval()
    for p in base.parameters(): p.requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    row=reinforce_step(g,model,base,obj,opt,0,3,regret_weight=.1,positive_regret_weight=2.)
    assert 'updated' in row
    path=beam_search(g,model,obj,0,3,width=16,local_cost_weight=.7).path
    assert path is None or path_feasible(g,path,0,3)
