"""TC-DSFNet architecture, losses and candidate selection (Eqs. 1-11)."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .splines import SplineCoupling

WINDOWS = {'single': ([60,20,2,1], [100,30,4,4]),
           'multi': ([60,20,2,1,0], [100,30,4,4,80])}


def features(y):
    if torch.any(y[..., 1] <= 0):
        raise ValueError('Bead height must be positive')
    w, h = y.unbind(-1)
    return torch.stack([w,h,w/h,w*h], -1)


def mlp(a, b, hidden):
    return nn.Sequential(nn.Linear(a,hidden), nn.SiLU(), nn.Linear(hidden,hidden),
                         nn.SiLU(), nn.Linear(hidden,b))


class TCDSFNet(nn.Module):
    def __init__(self, hidden=64, context=32, bins=8, layers=6, n_layers=10):
        super().__init__()
        self.config = dict(hidden=hidden, context=context, bins=bins, layers=layers, n_layers=n_layers)
        self.n_layers = n_layers
        self.encoder = mlp(4, context, hidden)
        self.domain_embedding = nn.Embedding(2, 8)
        self.adapters = nn.ModuleDict({d: nn.ModuleList([nn.Linear(8,2*context) for _ in range(layers)]) for d in WINDOWS})
        # Eq. (2): c_d^l = gamma_d^l * c + beta_d^l. Initialize adapters
        # at the identity (gamma=1, beta=0) without changing the equation.
        for stack in self.adapters.values():
            for adapter in stack:
                nn.init.zeros_(adapter.weight)
                with torch.no_grad():
                    adapter.bias[:context].fill_(1.)
                    adapter.bias[context:].zero_()
        # Fixed coupling partition plus alternating roll permutations ensures
        # every physical coordinate is transformed (including odd dimension 5).
        self.flows = nn.ModuleDict({d: nn.ModuleList([SplineCoupling(len(v[0]), context, hidden, bins, 0) for k in range(layers)]) for d,v in WINDOWS.items()})
        self.geometry = nn.ModuleDict({d:mlp(len(v[0]),2,hidden) for d,v in WINDOWS.items()})
        self.layer_embedding = nn.Embedding(n_layers, 8)
        self.deviation = mlp(5+8,2,hidden)
        self.register_buffer('feature_mean',torch.zeros(4))
        self.register_buffer('feature_scale',torch.ones(4))
        for d, (lo,hi) in WINDOWS.items():
            self.register_buffer(d+'_lower',torch.tensor(lo,dtype=torch.float))
            self.register_buffer(d+'_range',torch.tensor(np.array(hi)-lo,dtype=torch.float))
            self.register_buffer(d+'_ymean',torch.zeros(2))
            self.register_buffer(d+'_yscale',torch.ones(2))
        self.register_buffer('deviation_scale',torch.ones(2))

    @torch.no_grad()
    def fit_scalers(self, train_y, deviations=None):
        """Call with measured training rows only, after group splitting."""
        f = features(torch.cat(list(train_y.values()),0))
        self.feature_mean.copy_(f.mean(0))
        self.feature_scale.copy_(f.std(0,unbiased=False).clamp_min(1e-6))
        for d,y in train_y.items():
            getattr(self,d+'_ymean').copy_(y.mean(0))
            getattr(self,d+'_yscale').copy_(y.std(0,unbiased=False).clamp_min(1e-6))
        if deviations is not None:
            self.deviation_scale.copy_(deviations.reshape(-1,2).std(0,unbiased=False).clamp_min(1e-6))

    def normalize(self, x, domain):
        rho=(x-getattr(self,domain+'_lower'))/getattr(self,domain+'_range')
        if torch.any((rho < -1e-6) | (rho > 1+1e-6)):
            raise ValueError('Process parameter outside manuscript process window')
        return rho.clamp(1e-4,1-1e-4)

    def conditions(self, y, domain):
        c=self.encoder((features(y)-self.feature_mean)/self.feature_scale)
        e=self.domain_embedding.weight[0 if domain=='single' else 1]
        result=[]
        for adapter in self.adapters[domain]:
            gamma,beta=adapter(e).chunk(2,-1)
            result.append(gamma*c+beta)
        return result

    def transform(self, q, y, domain, inverse=False):
        y=y.expand(*q.shape[:-1],2)
        contexts=self.conditions(y,domain)
        logdet=torch.zeros(q.shape[:-1],device=q.device,dtype=q.dtype)
        indices=range(len(contexts)-1,-1,-1) if inverse else range(len(contexts))
        for i in indices:
            if inverse:
                q=torch.roll(q,shifts=-(1 if i%2==0 else 2),dims=-1)
            q,ld=self.flows[domain][i](q,contexts[i],inverse)
            if not inverse:
                q=torch.roll(q,shifts=(1 if i%2==0 else 2),dims=-1)
            logdet=logdet+ld
        return q,logdet

    def log_prob(self, x, y, domain):
        rho=self.normalize(x,domain)
        q=torch.logit(rho)
        z,ld=self.transform(q,y,domain)
        normal=-.5*(z*z+math.log(2*math.pi)).sum(-1)
        correction=torch.log(getattr(self,domain+'_range')*rho*(1-rho)).sum(-1)
        return normal+ld-correction

    def generate(self, y, domain, n=128, temperature=1., generator=None):
        if n < 1 or temperature <= 0:
            raise ValueError('Positive candidate count and temperature required')
        expanded=y[:,None,:].expand(-1,n,-1)
        z=torch.randn(*expanded.shape[:-1],len(WINDOWS[domain][0]),device=y.device,
                      dtype=y.dtype,generator=generator)*temperature
        q,_=self.transform(z,expanded,domain,inverse=True)
        rho=torch.sigmoid(q).clamp(1e-7,1-1e-7)
        return getattr(self,domain+'_lower')+getattr(self,domain+'_range')*rho

    def predict_geometry(self, rho, domain):
        return self.geometry[domain](rho)*getattr(self,domain+'_yscale')+getattr(self,domain+'_ymean')

    def predict_layers(self, rho):
        shape=rho.shape[:-1]
        r=rho.unsqueeze(-2).expand(*shape,self.n_layers,5)
        e=self.layer_embedding.weight.expand(*shape,self.n_layers,8)
        a=self.deviation(torch.cat([r,e],-1))*self.deviation_scale
        delta=a-a.mean(-2,keepdim=True)
        return self.predict_geometry(rho,'multi').unsqueeze(-2)+delta,delta

    def losses(self, batches, signs, layer_batch=None, weights=(10.,2.,.5,.5),
               layer_signs=(False,False), deviation_weight=1.):
        """Weighted NLL + standardized geometry/cycle/trend/layer losses.

        batches[d] = (physical x, geometry y, measured/interpolated weights).
        layer_batch = (full physical conditions [B,5], layer geometries [B,L,2]).
        signs[d] contains (parameter index, geometry index, bootstrap sign).
        """
        zero=self.feature_mean.new_zeros(())
        flow_num,weight_sum=zero,zero
        geo,cyc,trend,layer=zero,zero,zero,zero
        trend_violations=[]
        for d,(x,y,w) in batches.items():
            flow_num=flow_num-(self.log_prob(x,y,d)*w).sum()
            weight_sum=weight_sum+w.sum()
            rho=self.normalize(x,d).detach().requires_grad_(True)
            prediction=self.predict_geometry(rho,d)
            scale=getattr(self,d+'_yscale')
            geo=geo+(((prediction-y)/scale).abs().mean(-1)*w).sum()/w.sum()
            generated=self.generate(y,d,n=1)[:,0,:]
            cyc=cyc+((self.predict_geometry(self.normalize(generated,d),d)-y)/scale).abs().mean()
            for k,r,sign in signs.get(d,[]):
                jac=torch.autograd.grad(prediction[:,r].sum(),rho,create_graph=True,retain_graph=True)[0][:,k]
                trend_violations.append(F.relu(-float(sign)*jac).mean())
        if trend_violations:
            # Eq. (6): one mean over every retained stable relationship in C,
            # irrespective of domain.  Do not average per domain and then add.
            trend=torch.stack(trend_violations).mean()
        if layer_batch is not None:
            x,observed=layer_batch
            prediction,delta=self.predict_layers(self.normalize(x,'multi'))
            geo=geo+deviation_weight*((prediction-observed)/self.multi_yscale).abs().mean()
            terms=[]
            if layer_signs[0]:
                terms.append(F.relu(prediction[:,:-1,0]-prediction[:,1:,0]).mean())
            if layer_signs[1]:
                terms.append(F.relu(prediction[:,1:,1]-prediction[:,:-1,1]).mean())
            if terms:
                layer=sum(terms)
        flow=flow_num/weight_sum.clamp_min(1)
        lg,lc,lt,ll=weights
        total=flow+lg*geo+lc*cyc+lt*trend+ll*layer
        return total,dict(flow=flow,geometry=geo,cycle=cyc,trend=trend,layer=layer)

    def candidate_metrics(self, x, y, domain):
        rho=self.normalize(x,domain)
        eg=(((self.predict_geometry(rho,domain)-y)/getattr(self,domain+'_yscale'))**2).sum(-1)
        density=-self.log_prob(x,y,domain)/x.shape[-1]
        boundary=-torch.log((4*rho*(1-rho)+1e-8)/(1+1e-8)).mean(-1)
        smooth=torch.zeros_like(eg)
        if domain=='multi':
            _,delta=self.predict_layers(rho)
            smooth=(((delta[...,1:,:]-delta[...,:-1,:])/self.deviation_scale)**2).sum(-1).mean(-1)
        return torch.stack([eg,density,boundary,smooth],-1)


@torch.no_grad()
def calibrate(model, x_val, y_val, domain, candidates=128, seed=4321):
    """Validation-only temperature, metric quantiles and score weights.

    Common random numbers ensure all temperature candidates are evaluated
    using the same latent draws.
    """
    trials=[]
    for temp in (.5,.75,1.,1.25,1.5):
        gen=torch.Generator(device=y_val.device).manual_seed(seed)
        x=model.generate(y_val,domain,candidates,temp,gen)
        qlo,qhi=torch.quantile(x,.05,dim=1),torch.quantile(x,.95,dim=1)
        coverage=((x_val>=qlo)&(x_val<=qhi)).float().mean().item()
        width=((qhi-qlo)/getattr(model,domain+'_range')).mean().item()
        trials.append((abs(coverage-.9)+.05*width,temp,coverage,width))
    chosen=min(trials)
    x=model.generate(y_val,domain,candidates,chosen[1],torch.Generator(device=y_val.device).manual_seed(seed))
    metrics=model.candidate_metrics(x,y_val[:,None,:],domain)
    low=torch.quantile(metrics.flatten(0,1),.05,dim=0)
    high=torch.quantile(metrics.flatten(0,1),.95,dim=0)
    normalized=((metrics-low)/(high-low+1e-8)).clamp(0,1)
    grids=([.7,.2,.1,0],[.8,.1,.1,0],[.5,.3,.2,0]) if domain=='single' else ([.6,.2,.1,.1],[.7,.1,.1,.1],[.5,.2,.1,.2])
    scored=[]
    for w in grids:
        idx=(normalized*metrics.new_tensor(w)).sum(-1).argmin(-1)
        selected=x[torch.arange(len(x)),idx]
        err=((selected-x_val)/getattr(model,domain+'_range')).abs().mean().item()
        scored.append((err,w))
    return dict(temperature=chosen[1],coverage=chosen[2],width=chosen[3],
                quantile_low=low.tolist(),quantile_high=high.tolist(),weights=min(scored)[1],
                trials=[list(v) for v in trials],seed=seed)


@torch.no_grad()
def design(model, targets, domain, calibration, candidates=128, seed=123):
    x=model.generate(targets,domain,candidates,calibration['temperature'],
                     torch.Generator(device=targets.device).manual_seed(seed))
    metrics=model.candidate_metrics(x,targets[:,None,:],domain)
    lo=metrics.new_tensor(calibration['quantile_low'])
    hi=metrics.new_tensor(calibration['quantile_high'])
    scores=(((metrics-lo)/(hi-lo+1e-8)).clamp(0,1)*metrics.new_tensor(calibration['weights'])).sum(-1)
    idx=scores.argmin(-1)
    return dict(candidates=x,metrics=metrics,scores=scores,index=idx,selected=x[torch.arange(len(x)),idx])
