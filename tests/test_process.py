import numpy as np
import torch
import pytest
from gppaf.splines import rational_quadratic
from gppaf.process import TCDSFNet, WINDOWS
from gppaf.data import grouped_split,augment_training


@pytest.mark.parametrize('seed',[0,7,13])
def test_spline_inverse_and_autograd_jacobian(seed):
    torch.manual_seed(seed)
    x=(torch.rand(25,3,dtype=torch.double)*12-6).requires_grad_(True)
    raw=torch.randn(25,3,23,dtype=torch.double)*.5
    y,ld=rational_quadratic(x,raw)
    recovered,ild=rational_quadratic(y,raw,True)
    assert torch.max(abs(x-recovered))<1e-8
    assert torch.max(abs(ld+ild))<1e-8
    derivative=torch.autograd.grad(y.sum(),x)[0]
    assert torch.allclose(torch.log(derivative),ld,atol=1e-8)
    h=1e-5
    numeric=(rational_quadratic(x.detach()+h,raw)[0]-rational_quadratic(x.detach()-h,raw)[0])/(2*h)
    assert torch.allclose(numeric,derivative,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('domain',['single','multi'])
def test_full_flow_roundtrip_and_bounds(domain):
    torch.manual_seed(3);m=TCDSFNet(hidden=16,context=8).double()
    y=torch.tensor([[10.,2.5],[11.,2.]],dtype=torch.double)
    q=torch.randn(2,len(WINDOWS[domain][0]),dtype=torch.double)
    z,ld=m.transform(q,y,domain)
    r,ild=m.transform(z,y,domain,True)
    # Six alternating one-/two-position permutations: verify that no original
    # coordinate remains an identity passthrough across the entire flow.
    unpermuted=torch.roll(z,shifts=-9,dims=-1)
    assert torch.all((unpermuted-q).abs().max(0)[0]>1e-8)
    assert torch.allclose(q,r,atol=1e-8)
    assert torch.allclose(ld,-ild,atol=1e-8)
    generated=m.generate(y,domain,12)
    lo,hi=WINDOWS[domain]
    assert torch.all(generated>torch.tensor(lo)) and torch.all(generated<torch.tensor(hi))
    assert torch.isfinite(m.log_prob(generated,y[:,None,:],domain)).all()


def test_layer_zero_mean_and_training_gradients():
    m=TCDSFNet(hidden=16,context=8)
    y=torch.tensor([[10.,2.5],[11.,2.]])
    m.fit_scalers({'single':y,'multi':y})
    x=torch.tensor([[80.,25.,3.,2.5,40.],[90.,28.,3.5,3.,60.]])
    predicted,delta=m.predict_layers(m.normalize(x,'multi'))
    assert torch.allclose(delta.mean(1),torch.zeros(2,2),atol=1e-6)
    batch={'single':(x[:,:4],y,torch.ones(2)),'multi':(x,y,torch.ones(2))}
    loss,terms=m.losses(batch,{'single':[(0,0,1)],'multi':[(1,0,-1)]},(x,y[:,None,:].repeat(1,10,1)),layer_signs=(True,True))
    loss.backward()
    assert torch.isfinite(loss)
    for module in (m.encoder,m.flows['single'],m.flows['multi'],m.geometry,m.deviation):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())


def test_group_split_and_interpolation_parent_audit():
    groups=np.repeat(np.arange(30),3)
    split=grouped_split(groups)
    for a,b in [('train','test'),('train','validation'),('test','validation')]:
        assert not set(groups[split[a]])&set(groups[split[b]])
    x=np.arange(180).reshape(90,2);y=x/10
    train=split['train']
    ax,ay,w,parents=augment_training(x[train],y[train],seed=2)
    assert len(ax)==5*len(train)
    assert parents.max()<len(train)
    assert np.all(w[:len(train)]==1) and np.all(w[len(train):]==.3)
    assert np.allclose(ay,ax/10)


def test_domain_modulation_matches_equation_2():
    """Eq. (2) is gamma*c+beta, not a residual (1+gamma)*c+beta map."""
    torch.manual_seed(11)
    m=TCDSFNet(hidden=16,context=8,layers=1)
    y=torch.tensor([[10.0,2.5]])
    c=m.encoder((torch.stack((y[:,0],y[:,1],y[:,0]/y[:,1],y[:,0]*y[:,1]),-1)-m.feature_mean)/m.feature_scale)
    adapter=m.adapters['single'][0]
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.bias[:8].fill_(0.5)
        adapter.bias[8:].fill_(0.2)
    out=m.conditions(y,'single')[0]
    assert torch.allclose(out,0.5*c+0.2,atol=1e-7,rtol=0)


def test_trend_loss_uses_one_mean_over_all_retained_constraints():
    """Eq. (6) averages every retained relationship in C exactly once."""
    torch.manual_seed(17)
    m=TCDSFNet(hidden=16,context=8,layers=1)
    y=torch.tensor([[9.5,2.3],[10.5,2.6],[11.0,2.4]],dtype=torch.float32)
    xs=torch.tensor([[70.,24.,2.5,2.0],[80.,26.,3.0,3.0],[90.,28.,3.5,3.5]],dtype=torch.float32)
    xm=torch.tensor([[70.,24.,2.5,2.0,20.],[80.,26.,3.0,3.0,40.],[90.,28.,3.5,3.5,60.]],dtype=torch.float32)
    m.fit_scalers({'single':y,'multi':y})
    batches={'single':(xs,y,torch.ones(3)),'multi':(xm,y,torch.ones(3))}
    signs={'single':[(0,0,1)],'multi':[(0,0,1),(1,1,-1)]}
    _,terms=m.losses(batches,signs)

    expected=[]
    for d,x in [('single',xs),('multi',xm)]:
        rho=m.normalize(x,d).detach().requires_grad_(True)
        pred=m.predict_geometry(rho,d)
        for k,r,sign in signs[d]:
            jac=torch.autograd.grad(pred[:,r].sum(),rho,create_graph=True,retain_graph=True)[0][:,k]
            expected.append(torch.relu(-float(sign)*jac).mean())
    expected=torch.stack(expected).mean()
    assert torch.allclose(terms['trend'],expected,atol=1e-7,rtol=0)
