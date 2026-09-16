"""Reference process-design and path-planning baselines used by GPPAF benchmarks."""
import math
import numpy as np
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from .search import feasible_candidates,SearchResult,local_search
from .geometry import path_feasible


def classical_process(kind,x,y,seed=0):
    if kind=='RSM': model=make_pipeline(PolynomialFeatures(2),Ridge(alpha=1e-6))
    elif kind=='SVR': model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=10,epsilon=.05)))
    elif kind=='RF': model=RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=seed,n_jobs=1)
    else: raise ValueError('Unknown baseline '+kind)
    model.fit(x,y);return model


def classical_inverse(model,target,lower,upper,geometry_scale,candidates=32,seed=0):
    """Same bounded multistart forward-model search for RSM/SVR/RF.

    All baseline settings are explicit so a benchmark run can be repeated.
    """
    rng=np.random.RandomState(seed);lo=np.array(lower);span=np.array(upper)-lo
    result=[]
    def loss(rho): return float(np.sum(((model.predict((lo+span*rho)[None,:])[0]-target)/geometry_scale)**2))
    for initial in rng.uniform(size=(candidates,len(lo))):
        optimized=minimize(loss,initial,method='Powell',bounds=[(0,1)]*len(lo),options=dict(maxiter=80,xtol=1e-4,ftol=1e-5))
        rho=np.clip(optimized.x,0,1)
        result.append(lo+span*rho)
    return np.array(result)


class ConditionalVAE(nn.Module):
    def __init__(self,dimension,hidden=64,latent=8):
        super().__init__();self.latent=latent
        self.encoder=nn.Sequential(nn.Linear(dimension+2,hidden),nn.SiLU(),nn.Linear(hidden,2*latent))
        self.decoder=nn.Sequential(nn.Linear(latent+2,hidden),nn.SiLU(),nn.Linear(hidden,dimension))
    def loss(self,rho,geometry,beta=.01):
        mu,logvar=self.encoder(torch.cat([rho,geometry],-1)).chunk(2,-1)
        logvar=logvar.clamp(-12,8);z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        decoded=torch.sigmoid(self.decoder(torch.cat([z,geometry],-1)))
        return (decoded-rho).square().mean()+beta*.5*(mu.square()+logvar.exp()-1-logvar).mean()
    def sample(self,geometry,n,generator=None):
        g=geometry[:,None,:].expand(-1,n,-1)
        z=torch.randn(len(g),n,self.latent,device=g.device,generator=generator)
        return torch.sigmoid(self.decoder(torch.cat([z,g],-1)))


class AffineCINN(nn.Module):
    def __init__(self,dimension,hidden=64,layers=6):
        super().__init__();self.dimension=dimension
        self.nets=nn.ModuleList([nn.Sequential(nn.Linear(dimension+2,hidden),nn.SiLU(),nn.Linear(hidden,2*dimension)) for _ in range(layers)])
        self.register_buffer('masks',torch.tensor([[float(j%2==i%2) for j in range(dimension)] for i in range(layers)]))
    def transform(self,x,g,inverse=False):
        logdet=x.new_zeros(x.shape[:-1]);g=g.expand(*x.shape[:-1],2)
        for i in (range(len(self.nets)-1,-1,-1) if inverse else range(len(self.nets))):
            mask=self.masks[i]
            s,t=self.nets[i](torch.cat([x*mask,g],-1)).chunk(2,-1)
            s=2*torch.tanh(s)*(1-mask);t=t*(1-mask)
            x=(x-t)*torch.exp(-s) if inverse else x*torch.exp(s)+t
            logdet=logdet+(-s if inverse else s).sum(-1)
        return x,logdet
    def loss(self,rho,g):
        rho=rho.clamp(1e-4,1-1e-4);q=torch.logit(rho);z,ld=self.transform(q,g)
        logp=-.5*(z.square()+math.log(2*math.pi)).sum(-1)+ld-torch.log(rho*(1-rho)).sum(-1)
        return -logp.mean()
    def sample(self,g,n,generator=None):
        expanded=g[:,None,:].expand(-1,n,-1)
        z=torch.randn(len(g),n,self.dimension,device=g.device,generator=generator)
        return torch.sigmoid(self.transform(z,expanded,True)[0])


def nearest_neighbor(graph,objective,start,end):
    path=[start]
    while len(path)<len(graph.points):
        choices=feasible_candidates(graph,path,end)
        if not choices: return SearchResult(None,'search_failed',math.inf)
        path.append(min(choices,key=lambda j:(np.linalg.norm(graph.points[j]-graph.points[path[-1]]),j)))
    return SearchResult(path,'feasible',objective(graph,path))


def constrained_aco(graph,objective,start,end,ants=12,iterations=10,seed=0,evaporation=.2):
    rng=np.random.RandomState(seed);pheromone=np.ones(len(graph.edge_index));best=None;best_cost=math.inf
    for _ in range(iterations):
        successful=[]
        for _ in range(ants):
            path=[start]
            while len(path)<len(graph.points):
                choices=feasible_candidates(graph,path,end)
                if not choices: break
                ids=[graph.lookup[(path[-1],j)] for j in choices]
                weights=pheromone[ids]/np.maximum([graph.segments[(path[-1],j)].length for j in choices],1e-8)**2
                path.append(int(rng.choice(choices,p=weights/weights.sum())))
            if path_feasible(graph,path,start,end):
                cost=objective(graph,path);successful.append((path,cost))
                if cost<best_cost: best,best_cost=path,cost
        pheromone*=1-evaporation
        for path,cost in successful:
            for edge in zip(path[:-1],path[1:]): pheromone[graph.lookup[edge]]+=1/max(cost,1e-8)
    return SearchResult(best,'feasible' if best is not None else 'search_failed',best_cost)
