"""Relation-aware node/edge Transformer and second-order decoder, Eqs. 28-33."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import angle


def typed_projection(modules,x,types):
    result=torch.zeros(len(x),modules[0].out_features,device=x.device,dtype=x.dtype)
    for t,m in enumerate(modules):
        mask=types==t
        if mask.any(): result[mask]=m(x[mask])
    return result


class RelationLayer(nn.Module):
    def __init__(self,hidden,heads):
        super().__init__()
        if hidden%heads: raise ValueError('Hidden width must be divisible by heads')
        self.hidden=hidden;self.heads=heads;self.dk=hidden//heads
        for name in ('q','kn','vn','ke','ve'):
            setattr(self,name,nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(3)]))
        self.bias=nn.Parameter(torch.zeros(3,heads))
        self.node_out=nn.Linear(hidden,hidden)
        self.node_norm=nn.LayerNorm(hidden)
        self.ff=nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))
        self.ffnorm=nn.LayerNorm(hidden)
        self.edge_ff=nn.Sequential(nn.Linear(3*hidden,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))
        self.edge_norm=nn.LayerNorm(hidden)

    def forward(self,h,g,edges,nt,rt):
        src,dst=edges[:,0],edges[:,1]
        q=typed_projection(self.q,h,nt)[src].view(-1,self.heads,self.dk)
        k=(typed_projection(self.kn,h,nt)[dst]+typed_projection(self.ke,g,rt)).view(-1,self.heads,self.dk)
        v=(typed_projection(self.vn,h,nt)[dst]+typed_projection(self.ve,g,rt)).view(-1,self.heads,self.dk)
        logits=(q*k).sum(-1)/math.sqrt(self.dk)+self.bias[rt]
        # Segment softmax by source node; sparse edge storage is preserved.
        alpha=torch.zeros_like(logits)
        for i in range(len(h)):
            mask=src==i
            if mask.any(): alpha[mask]=torch.softmax(logits[mask],dim=0)
        messages=(alpha.unsqueeze(-1)*v).reshape(-1,self.hidden)
        agg=torch.zeros_like(h).index_add(0,src,messages)
        h1=self.node_norm(h+self.node_out(agg))
        h1=self.ffnorm(h1+self.ff(h1))
        g1=self.edge_norm(g+self.edge_ff(torch.cat([h1[src],h1[dst],g],-1)))
        return h1,g1


class PAHGTS(nn.Module):
    def __init__(self,hidden=64,heads=4,layers=3,clip=10.):
        super().__init__()
        self.config=dict(hidden=hidden,heads=heads,layers=layers,clip=clip)
        self.hidden=hidden;self.clip=clip
        self.node_input=nn.ModuleList([nn.Linear(8,hidden) for _ in range(3)])
        self.edge_input=nn.ModuleList([nn.Linear(7,hidden) for _ in range(3)])
        self.nt_embedding=nn.Embedding(3,hidden);self.rt_embedding=nn.Embedding(3,hidden)
        self.encoder=nn.ModuleList([RelationLayer(hidden,heads) for _ in range(layers)])
        self.priority=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.regret=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.start=nn.Parameter(torch.zeros(hidden))
        self.context=nn.Linear(4*hidden+6,hidden)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.key_node=nn.Linear(hidden,hidden,bias=False)
        self.key_edge=nn.Linear(hidden,hidden,bias=False)
        self.process_raw=nn.Parameter(torch.full((8,),-1.))
        self.prior_raw=nn.Parameter(torch.tensor(0.))

    def encode(self,graph):
        device=self.start.device; dtype=self.start.dtype
        nf=torch.as_tensor(graph.node_features,device=device,dtype=dtype)
        ef=torch.as_tensor(graph.edge_features,device=device,dtype=dtype)
        nt=torch.as_tensor(graph.node_types,device=device,dtype=torch.long)
        rt=torch.as_tensor(graph.relations,device=device,dtype=torch.long)
        edges=torch.as_tensor(graph.edge_index,device=device,dtype=torch.long)
        h=typed_projection(self.node_input,nf,nt)+self.nt_embedding(nt)
        g=typed_projection(self.edge_input,ef,rt)+self.rt_embedding(rt)
        for layer in self.encoder: h,g=layer(h,g,edges,nt,rt)
        joint=torch.cat([h[edges[:,0]],h[edges[:,1]],g],-1)
        return h,g,F.softplus(self.priority(joint).squeeze(-1))+1e-8,torch.sigmoid(self.regret(joint).squeeze(-1))

    def scores(self,graph,path,candidates,objective,encoded):
        h,g,eta,_=encoded
        remaining=sorted(set(range(len(h)))-set(path))
        rem=h[remaining].mean(0) if remaining else torch.zeros_like(h[0])
        prev=h[path[-2]] if len(path)>1 else self.start
        costs=objective.metrics(graph,path)
        status=h.new_tensor([len(path)/len(h),*(costs/np.array(objective.scales))])
        context=self.context(torch.cat([h.mean(0),rem,prev,h[path[-1]],status]))
        edge_ids=[graph.lookup[(path[-1],j)] for j in candidates]
        process=[]
        for j,k in zip(candidates,edge_ids):
            turn=angle(graph.points[path[-1]]-graph.points[path[-2]],graph.points[j]-graph.points[path[-1]])/math.pi if len(path)>1 else 0.
            delta=objective.metrics(graph,path+[j])-costs
            process.append([graph.edge_features[k,0],turn,*(delta[2:]/np.array(objective.scales)[2:]),*np.eye(3)[graph.relations[k]]])
        psi=h.new_tensor(np.array(process))
        keys=self.key_node(h[candidates])+self.key_edge(g[edge_ids])
        raw=(keys*self.query(context)).sum(-1)/math.sqrt(self.hidden)
        raw=raw+F.softplus(self.prior_raw)*torch.log(eta[edge_ids])-(psi*F.softplus(self.process_raw)).sum(-1)
        return self.clip*torch.tanh(raw)
