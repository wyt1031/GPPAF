import numpy as np
import torch
from gppaf.baselines import ConditionalVAE,AffineCINN
from gppaf.evaluation import stress_metrics,point_cloud_metrics,paired_comparison
from gppaf.geometry import build_graph
from gppaf.layers import LayerConfig,within_layer,interlayer_edge
from shapely.geometry import Polygon


def test_affine_inverse_and_vae_bounds():
    torch.manual_seed(2)
    for dim in (4,5):
        model=AffineCINN(dim,hidden=16)
        x=torch.randn(10,dim);g=torch.randn(10,2)
        z,ld=model.transform(x,g);rec,ild=model.transform(z,g,True)
        assert torch.allclose(x,rec,atol=2e-6) and torch.allclose(ld,-ild,atol=2e-6)
        for baseline in (model,ConditionalVAE(dim,hidden=16)):
            loss=baseline.loss(torch.rand(10,dim),g);loss.backward()
            sample=baseline.sample(g,5)
            assert sample.shape==(10,5,dim) and torch.all((sample>0)&(sample<1))


def test_global_stress_and_height_scaling():
    result=stress_metrics([np.array([0,1,2]),np.array([2,4])])
    assert result[0]['high_stress_fraction']==1/3
    a=np.array([[0,0,0],[1,0,1],[2,0,2.]])
    b=np.array([[0,0,2],[1,0,3],[2,0,4.]])
    result=point_cloud_metrics([a,b])
    assert result['global_height_min']==0 and result['global_height_max']==4
    assert np.isclose(result['metrics'][0]['outlier_fraction'],2/3)


def test_holm_and_paired_identity():
    r=paired_comparison([1,2,3,4],{'same':[1,2,3,4],'higher':[2,3,4,5]},resamples=100)
    assert r['same']['p_holm']==1 and r['same']['paired_dominance']==.5
    assert r['higher']['p_holm']>=r['higher']['p_raw']


def test_sampling_contains_all_heterogeneous_node_types():
    g=build_graph([(0,0),(8,0),(8,8),(0,8)],holes=[[(3,3),(5,3),(5,5),(3,5)]],bead_width=1,phase=(.5,.5),max_scale=2.1)
    assert set(g.node_types)=={0,1,2}
    assert set(g.relations)=={0,1,2}


def test_tilt_is_measured_from_vertical():
    safe=Polygon([(-1,-1),(10,-1),(10,5),(-1,5)])
    cfg=LayerConfig(retained_states=2,max_tilt_from_vertical=10,max_interlayer=20)
    a=within_layer([np.array([[0.,0],[1,0]])],safe,0,cfg)[0]
    b=within_layer([np.array([[8.,0],[9,0]])],safe,1,cfg)[0]
    assert interlayer_edge(a,b,safe,cfg) is None
