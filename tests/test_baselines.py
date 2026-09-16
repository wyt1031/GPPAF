"""Execution coverage for the comparison methods used in the manuscript."""
import numpy as np
import torch

from gppaf.baselines import (
    classical_process, classical_inverse, ConditionalVAE, AffineCINN,
    nearest_neighbor, constrained_aco,
)
from gppaf.geometry import build_graph
from gppaf.objectives import Objective


def process_fixture():
    x=np.array([
        [60.,22.,2.,1.], [70.,24.,2.5,1.8], [80.,26.,3.,2.5],
        [90.,28.,3.5,3.2], [100.,30.,4.,4.], [85.,25.,3.2,2.8],
    ])
    y=np.column_stack([8.+0.03*x[:,0]-0.04*x[:,1]+0.2*x[:,3],
                       1.5+0.01*x[:,0]-0.02*x[:,1]-0.05*x[:,3]])
    return x,y


def test_classical_process_baselines_fit_and_inverse():
    x,y=process_fixture(); lo=np.array([60.,20.,2.,1.]); hi=np.array([100.,30.,4.,4.])
    for kind in ('RSM','SVR','RF'):
        model=classical_process(kind,x,y,seed=7)
        pred=model.predict(x)
        assert pred.shape==y.shape and np.isfinite(pred).all()
        inv=classical_inverse(model,y[2],lo,hi,np.maximum(y.std(0),1e-6),candidates=1,seed=7)
        assert inv.shape==(1,4) and np.isfinite(inv).all()
        assert np.all(inv>=lo) and np.all(inv<=hi)


def test_generative_process_baselines_train_and_sample():
    torch.manual_seed(7)
    rho=torch.rand(8,4); geometry=torch.rand(8,2)
    for model in (ConditionalVAE(4,hidden=8,latent=3),AffineCINN(4,hidden=8,layers=2)):
        loss=model.loss(rho,geometry); loss.backward()
        assert torch.isfinite(loss)
        sample=model.sample(geometry[:2],3,torch.Generator().manual_seed(9))
        assert sample.shape==(2,3,4) and torch.isfinite(sample).all()
        assert torch.all((sample>0)&(sample<1))


def test_path_baselines_return_declared_status():
    points=[(0,0),(1,0),(0,1),(1,1)]
    graph=build_graph([(-.5,-.5),(1.5,-.5),(1.5,1.5),(-.5,1.5)],
                      bead_width=.4,points=points,k_neighbors=4,max_scale=4)
    objective=Objective(scales=(4.,1.,4.,4.,4.))
    nn=nearest_neighbor(graph,objective,0,3)
    aco=constrained_aco(graph,objective,0,3,ants=4,iterations=2,seed=3)
    assert nn.status in {'feasible','search_failed'}
    assert aco.status in {'feasible','search_failed'}
    assert nn.path is not None or np.isinf(nn.objective)
    assert aco.path is not None or np.isinf(aco.objective)
