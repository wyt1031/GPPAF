"""Hard-constrained decoding, beam search and feasible local search (34-37)."""
import math
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .geometry import path_feasible, angle
from .objectives import mst_lower_bound


@dataclass
class SearchResult:
    path: object
    status: str
    objective: float
    expanded: int = 0
    reason: str = ''


def residual_ok(graph,nodes,current,end):
    """Necessary directed reachability/degree conditions, not a Hamiltonian proof."""
    nodes=set(nodes)
    if nodes=={end}: return current==end
    if current==end: return False
    edges=[(i,j) for i,j in graph.edge_index if i in nodes and j in nodes and i!=end and j!=current]
    incoming={i:[] for i in nodes}; outgoing={i:[] for i in nodes}
    for i,j in edges: outgoing[i].append(j); incoming[j].append(i)
    for i in nodes:
        if i!=end and not outgoing[i]: return False
        if i!=current and not incoming[i]: return False
    def reachable(start,adj):
        seen={start}; todo=[start]
        while todo:
            for j in adj[todo.pop()]:
                if j not in seen: seen.add(j);todo.append(j)
        return seen
    return reachable(current,outgoing)==nodes and reachable(end,incoming)==nodes


def feasible_candidates(graph,path,end):
    remaining=set(range(len(graph.points)))-set(path)-{end}
    candidates=sorted(remaining) if remaining else [end]
    result=[]
    for j in candidates:
        if j in path or (path[-1],j) not in graph.lookup: continue
        new=path+[j]
        if not path_feasible(graph,new,complete=False): continue
        nodes=(remaining-{j})|{j,end}
        if residual_ok(graph,nodes,j,end): result.append(j)
    return result


@torch.no_grad()
def beam_search(graph,model,objective,start=0,end=None,width=32,lambda_cost=1.,lambda_bound=.1,refine=True,local_cost_weight=1.,local_rounds=2,local_top_edges=8):
    """Return a fully checked complete path, or explicit sparse-search failure."""
    n=len(graph.points); end=n-1 if end is None else end
    if not 0<=start<n or not 0<=end<n or start==end or width<1: raise ValueError('Invalid endpoints/beam width')
    encoded=model.encode(graph) if model is not None else None
    beams=[([start],0.,0.)]; expanded=0
    for _ in range(n-1):
        next_beams=[]
        for path,logp,_ in beams:
            choices=feasible_candidates(graph,path,end)
            if not choices: continue
            if model is None:
                logits=torch.tensor([-objective(graph,path+[j]) for j in choices])
            else: logits=model.scores(graph,path,choices,objective,encoded)
            log_probs=F.log_softmax(logits,-1).cpu().numpy()
            for j,lp in zip(choices,log_probs):
                new=path+[j]; expanded+=1
                bound=0. if len(new)==n else mst_lower_bound(graph,set(range(n))-set(new),j,end)
                if not math.isfinite(bound): continue
                score=logp+float(lp)-lambda_cost*objective(graph,new)-lambda_bound*bound/objective.scales[0]
                next_beams.append((new,logp+float(lp),score))
        if not next_beams:
            return SearchResult(None,'search_failed',math.inf,expanded,'No complete path retained; does not prove geometric infeasibility.')
        next_beams.sort(key=lambda v:(-v[2],v[0]))
        beams=next_beams[:width]
    complete=[p for p,_,_ in beams if path_feasible(graph,p,start,end)]
    if not complete: return SearchResult(None,'search_failed',math.inf,expanded,'Final independent feasibility check failed')
    best=min(complete,key=lambda p:objective(graph,p))
    regret=encoded[3].cpu().numpy() if encoded is not None else None
    if refine:
        best=local_search(graph,best,objective,regret=regret,rounds=local_rounds,top_edges=local_top_edges,local_cost_weight=local_cost_weight)
    return SearchResult(best,'feasible',objective(graph,best),expanded)


def neighborhood(path,edge=None):
    """Fixed-endpoint 2-opt and Or-opt(1,2,3), restricted to a removed edge."""
    n=len(path); original_edges=set(zip(path[:-1],path[1:])); seen=set()
    def allowed(candidate):
        key=tuple(candidate)
        if key in seen or key==tuple(path): return False
        seen.add(key)
        return edge is None or edge not in set(zip(candidate[:-1],candidate[1:]))
    for i in range(1,n-2):
        for j in range(i+1,n-1):
            candidate=path[:i]+path[i:j+1][::-1]+path[j+1:]
            if allowed(candidate): yield candidate
    for size in (1,2,3):
        for i in range(1,n-size):
            chain=path[i:i+size]; rest=path[:i]+path[i+size:]
            for j in range(1,len(rest)):
                candidate=rest[:j]+chain+rest[j:]
                if allowed(candidate): yield candidate


def local_search(graph,path,objective,regret=None,rounds=2,top_edges=8,local_cost_weight=1.):
    path=list(path)
    if not path_feasible(graph,path): raise ValueError('Local search requires a complete feasible seed')
    for _ in range(rounds):
        costs=[]
        for k,(i,j) in enumerate(zip(path[:-1],path[1:])):
            eid=graph.lookup[(i,j)]
            pred=0. if regret is None else regret[eid]
            length=graph.segments[(i,j)].length/objective.scales[0]
            turn=0.
            if k>0: turn+=angle(graph.points[i]-graph.points[path[k-1]],graph.points[j]-graph.points[i])
            if k+2<len(path): turn+=angle(graph.points[j]-graph.points[i],graph.points[path[k+2]]-graph.points[j])
            # Eq. 36 local heat/boundary/spacing contributions obtained from prefixes.
            delta=objective.metrics(graph,path[:k+2])-objective.metrics(graph,path[:k+1])
            local=objective.weights[0]*length+objective.weights[1]*turn/objective.scales[1]
            local+=float(np.dot(delta[2:]/np.array(objective.scales)[2:],objective.weights[2:]))
            costs.append((float(pred)+float(local_cost_weight)*local,(i,j)))
        best=list(path); best_cost=objective(graph,path)
        for _,edge in sorted(costs,reverse=True)[:top_edges]:
            for candidate in neighborhood(path,edge):
                if path_feasible(graph,candidate,path[0],path[-1]):
                    cost=objective(graph,candidate)
                    if cost<best_cost-1e-10: best,best_cost=candidate,cost
        if best==path: break
        path=best
    return path


def regret_labels(graph,path,objective,epsilon=1e-8):
    """Independent exhaustive feasible neighborhood oracle, Eq. (37).

    Prediction values never enter the labels. Only visited directed edges are
    supervised; unvisited edges are excluded from the loss.
    """
    base=objective(graph,path); labels=[]; ids=[]
    for edge in zip(path[:-1],path[1:]):
        best=base
        for candidate in neighborhood(path,edge):
            if path_feasible(graph,candidate,path[0],path[-1]): best=min(best,objective(graph,candidate))
        ids.append(graph.lookup[edge]); labels.append(float(base-best>epsilon))
    return ids,labels


def rollout(graph,model,objective,start,end,sample=True):
    encoded=model.encode(graph);path=[start];logs=[]
    for _ in range(len(graph.points)-1):
        choices=feasible_candidates(graph,path,end)
        if not choices: return None,logs,encoded
        logits=model.scores(graph,path,choices,objective,encoded)
        distribution=torch.distributions.Categorical(logits=logits)
        index=distribution.sample() if sample else logits.argmax()
        logs.append(distribution.log_prob(index));path.append(choices[int(index)])
    return (path if path_feasible(graph,path,start,end) else None),logs,encoded


def reinforce_step(graph,model,baseline,objective,optimizer,start,end,regret_weight=.1,positive_regret_weight=1.):
    """Feasible rollouts only; frozen independent greedy baseline is detached.

    Failure counts must be reported by the training driver. Neither a partial
    rollout nor a failed greedy baseline is represented as a feasible sample.
    """
    path,logs,encoded=rollout(graph,model,objective,start,end,True)
    with torch.no_grad(): base,_,_=rollout(graph,baseline,objective,start,end,False)
    if path is None or base is None:
        return dict(updated=False,sampled_failed=path is None,baseline_failed=base is None)
    cost=objective(graph,path); base_cost=objective(graph,base)
    policy=(cost-base_cost)*torch.stack(logs).sum()
    ids,labels=regret_labels(graph,path,objective)
    target=encoded[3].new_tensor(labels)
    if positive_regret_weight<=0: raise ValueError('positive_regret_weight must be positive')
    raw_regret=F.binary_cross_entropy(encoded[3][ids],target,reduction='none')
    class_weight=torch.where(target>0.5,target.new_tensor(float(positive_regret_weight)),target.new_tensor(1.0))
    loss_regret=(raw_regret*class_weight).mean()
    loss=policy+regret_weight*loss_regret
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    return dict(updated=True,loss=float(loss.detach()),objective=cost,baseline=base_cost,
                policy=float(policy.detach()),regret=float(loss_regret.detach()),positive_labels=sum(labels))
