"""Geometry-labeled within-layer DP and retained-state interlayer DP (41-45)."""
from dataclasses import dataclass, field
import math
import numpy as np
from shapely.geometry import LineString, Point
from .geometry import unit, allowed_intersection


@dataclass
class LayerConfig:
    cut_count: int = 4
    retained_states: int = 16
    endpoint_diversity: float = .25
    max_connection: float = 10.
    max_interlayer: float = 10.
    max_tilt_from_vertical: float = 80.
    distance_scale: float = 1.
    interlayer_scale: float = 1.
    weights: tuple = (.7,.2,.1)
    heat_radius: float = 2.
    cooling_time: float = 40.
    interlayer_dwell: float = 40.
    max_labels: int = 200000


@dataclass
class Traversal:
    component: int
    cut: int
    direction: int
    points: np.ndarray
    cost: float = 0.

    @property
    def entry(self): return self.points[0]
    @property
    def exit(self): return self.points[-1]
    @property
    def tin(self): return unit(self.points[1]-self.points[0])
    @property
    def tout(self): return unit(self.points[-1]-self.points[-2])


@dataclass
class LayerState:
    traversals: list
    connections: list
    cost: float
    points: np.ndarray
    z: float

    @property
    def entry(self): return self.traversals[0].entry
    @property
    def exit(self): return self.traversals[-1].exit
    @property
    def tin(self): return self.traversals[0].tin
    @property
    def tout(self): return self.traversals[-1].tout


def entry_exit_states(points,component,cut_count=4):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2: raise ValueError('Each component needs >=2 planar points')
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]
    if len(p)<2: raise ValueError('Zero-length component')
    line=LineString(p)
    if not line.is_simple: raise ValueError('Self-crossing input component')
    if np.linalg.norm(p[0]-p[-1])>1e-8:
        return [Traversal(component,0,s,p.copy() if s==1 else p[::-1].copy()) for s in (1,-1)]
    cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    result=[]
    for k in range(cut_count):
        distance=line.length*k/cut_count
        j=min(np.searchsorted(cumulative,distance,side='right')-1,len(p)-2)
        cut=np.array(line.interpolate(distance).coords[0])
        ordered=np.vstack([cut,p[j+1:],p[1:j+1],cut])
        ordered=ordered[np.r_[True,np.linalg.norm(np.diff(ordered,axis=0),axis=1)>1e-9]]
        for s in (1,-1): result.append(Traversal(component,k,s,ordered.copy() if s==1 else ordered[::-1].copy()))
    return result


def direction_cost(exit,entry,tout,tin):
    delta=np.linalg.norm(entry-exit)
    if delta<=1e-8: return float((1-np.dot(tout,tin))/2)
    direction=(entry-exit)/delta
    return float((2-np.dot(tout,direction)-np.dot(direction,tin))/4)


def local_heat(distance,cfg,dwell=0.):
    """Local thermal accumulation term H used by HELS-DP.

    The same Gaussian spatial decay and exponential cooling kernel used by the
    path heat term is evaluated for each candidate connection.
    """
    return math.exp(-distance**2/(2*cfg.heat_radius**2)-dwell/cfg.cooling_time)


def connection(a,b,safe,components,cfg):
    distance=float(np.linalg.norm(b.entry-a.exit))
    if distance>cfg.max_connection: return None
    line=LineString([a.exit,b.entry])
    if distance<=1e-8:
        if not safe.buffer(1e-8).covers(Point(a.exit)): return None
    else:
        if not safe.buffer(1e-8).covers(line): return None
        for component in components:
            if not allowed_intersection(line,component,[a.exit,b.entry]): return None
    w=cfg.weights
    cost=w[0]*distance/cfg.distance_scale+w[1]*direction_cost(a.exit,b.entry,a.tout,b.tin)+w[2]*local_heat(distance,cfg)
    return line,float(cost)


def assemble(traversals):
    values=[traversals[0].points]
    for q in traversals[1:]: values.append(q.points)
    p=np.vstack(values)
    return p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]


def within_layer(components,safe,z=0.,config=None):
    """Exact geometry-label enumeration before endpoint-state truncation.

    No cost-only pruning across different connector geometries. max_labels is
    an explicit resource guard: exceeding it raises, never silently prunes.
    """
    cfg=config or LayerConfig()
    if min(cfg.cut_count,cfg.retained_states,cfg.max_labels)<1: raise ValueError('Positive state counts required')
    if min(cfg.weights)<0 or not np.isclose(sum(cfg.weights),1): raise ValueError('Connection weights must sum to one')
    if min(cfg.distance_scale,cfg.interlayer_scale,cfg.heat_radius,cfg.cooling_time)<=0: raise ValueError('Invalid scales')
    lines=[LineString(c) for c in components]
    if not lines: return []
    if any(not safe.buffer(1e-8).covers(c) for c in lines): return []
    for i,a in enumerate(lines):
        for b in lines[i+1:]:
            contacts=[p for p in (a.coords[0],a.coords[-1]) if Point(p).distance(Point(b.coords[0]))<1e-8 or Point(p).distance(Point(b.coords[-1]))<1e-8]
            if not allowed_intersection(a,b,contacts): return []
    states=[q for i,c in enumerate(components) for q in entry_exit_states(c,i,cfg.cut_count)]
    edges={}
    for i,a in enumerate(states):
        for j,b in enumerate(states):
            if a.component!=b.component:
                item=connection(a,b,safe,lines,cfg)
                if item is not None: edges[(i,j)]=item
    labels=[(1<<q.component,[i],[],q.cost) for i,q in enumerate(states)]
    seen_count=len(labels)
    for _ in range(len(components)-1):
        next_labels=[]
        for mask,order,segments,cost in labels:
            for j,q in enumerate(states):
                if mask & (1<<q.component) or (order[-1],j) not in edges: continue
                segment,increment=edges[(order[-1],j)]
                contact=states[order[-1]].exit
                valid=True
                if segment.length>1e-8:
                    for old in segments:
                        if old.length>1e-8 and not allowed_intersection(segment,old,[contact]): valid=False;break
                if not valid: continue
                next_labels.append((mask|(1<<q.component),order+[j],segments+[segment],cost+increment+q.cost))
                seen_count+=1
                if seen_count>cfg.max_labels: raise RuntimeError('Geometry-label budget exceeded; reduce components/cuts or explicitly raise max_labels')
        labels=next_labels
        if not labels: return []
    results=[]
    for _,order,segments,cost in labels:
        traversal=[states[i] for i in order]
        results.append(LayerState(traversal,segments,float(cost),assemble(traversal),float(z)))
    results.sort(key=lambda s:s.cost)
    diverse=[];deferred=[]
    for s in results:
        if all(max(np.linalg.norm(s.entry-t.entry),np.linalg.norm(s.exit-t.exit))>=cfg.endpoint_diversity for t in diverse):
            diverse.append(s)
        else: deferred.append(s)
        if len(diverse)>=cfg.retained_states: break
    if len(diverse)<cfg.retained_states:
        diverse.extend(deferred[:cfg.retained_states-len(diverse)])
    return diverse


def interlayer_edge(a,b,safe,cfg):
    if b.z<=a.z: raise ValueError('Layer heights must strictly increase')
    planar=float(np.linalg.norm(b.entry-a.exit)); dz=b.z-a.z
    distance=math.hypot(planar,dz)
    if distance>cfg.max_interlayer or math.degrees(math.atan2(planar,dz))>cfg.max_tilt_from_vertical: return None
    # Prismatic allowable motion domain: the complete XY projection must be in
    # this domain. Strictly increasing z confines each connector to its slab.
    projected=Point(a.exit) if planar<=1e-8 else LineString([a.exit,b.entry])
    if not safe.buffer(1e-8).covers(projected): return None
    cost=cfg.weights[0]*distance/cfg.interlayer_scale+cfg.weights[1]*direction_cost(a.exit,b.entry,a.tout,b.tin)+cfg.weights[2]*local_heat(planar,cfg,cfg.interlayer_dwell)
    return float(cost),np.array([[*a.exit,a.z],[*b.entry,b.z]])


def connect_layers(layer_states,safe,config=None):
    """Optimal sequence over the supplied retained states, with full backtracking."""
    cfg=config or LayerConfig()
    if not layer_states or any(not s for s in layer_states):
        return dict(status='infeasible',reason='An input layer has no feasible complete state',points=None)
    costs=np.array([s.cost for s in layer_states[0]])
    predecessors=[];edge_cache=[]
    for left,right in zip(layer_states[:-1],layer_states[1:]):
        nxt=np.full(len(right),np.inf); pred=np.full(len(right),-1,dtype=int); cache={}
        for j,b in enumerate(right):
            for i,a in enumerate(left):
                edge=interlayer_edge(a,b,safe,cfg)
                if edge is None or not np.isfinite(costs[i]): continue
                value=costs[i]+edge[0]+b.cost
                if value<nxt[j]: nxt[j]=value;pred[j]=i;cache[(i,j)]=edge[1]
        if not np.isfinite(nxt).any(): return dict(status='infeasible',reason='No feasible interlayer transition',points=None)
        costs=nxt;predecessors.append(pred);edge_cache.append(cache)
    selected=[int(costs.argmin())]
    for pred in predecessors[::-1]: selected.append(int(pred[selected[-1]]))
    selected=selected[::-1];chosen=[s[i] for s,i in zip(layer_states,selected)]
    p=[]
    for s in chosen: p.extend(np.column_stack([s.points,np.full(len(s.points),s.z)]))
    p=np.array(p);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]
    connectors=[edge_cache[k][(selected[k],selected[k+1])].tolist() for k in range(len(chosen)-1)]
    return dict(status='feasible',points=p,cost=float(costs.min()),state_indices=selected,
                interlayer_connections=connectors,optimality='Exact over retained layer states; truncation may exclude a better global solution')
