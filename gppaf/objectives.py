"""Path objective and conservative graph lower bound, Eqs. (20)-(23),(35)."""
from dataclasses import dataclass
import math
import numpy as np
from shapely.geometry import Point
from .geometry import angle


@dataclass
class Objective:
    weights: tuple = (.35,.2,.15,.15,.15)
    scales: tuple = (1.,1.,1.,1.,1.)
    speed: float = 5.  # mm/s; 30 cm/min = 5 mm/s
    heat_radius: float = 2.
    cooling_time: float = 40.

    def __post_init__(self):
        if len(self.weights)!=5 or len(self.scales)!=5 or min(self.weights)<0 or not np.isclose(sum(self.weights),1):
            raise ValueError('Five nonnegative objective weights must sum to one')
        if min(self.scales)<=0 or min(self.speed,self.heat_radius,self.cooling_time)<=0:
            raise ValueError('Scales and process constants must be positive')

    def metrics(self,graph,path):
        if len(path)<2: return np.zeros(5)
        p=graph.points[path]; diff=np.diff(p,axis=0); lengths=np.linalg.norm(diff,axis=1)
        turns=sum(angle(a,b) for a,b in zip(diff[:-1],diff[1:]))/max((len(graph.points)-2)*math.pi,1)
        B,U=0.,0.; segments=[]
        for i,j in zip(path[:-1],path[1:]):
            k=graph.lookup[(i,j)]; relation=graph.relations[k]
            segment=graph.segments[(i,j)]; segments.append(segment)
            if relation==1:
                B+=abs(Point((graph.points[i]+graph.points[j])/2).distance(graph.region.boundary)-graph.bead_width/2)/graph.spacing
            elif relation==0:
                U+=abs(segment.length-graph.spacing)/graph.spacing
        times=np.r_[0,np.cumsum(lengths[:-1])]/self.speed
        heat=0.
        for a in range(len(segments)-2):
            for b in range(a+2,len(segments)):
                distance=segments[a].distance(segments[b])
                heat+=math.exp(-distance**2/(2*self.heat_radius**2)-(times[b]-times[a])/self.cooling_time)
        return np.array([lengths.sum(),turns,B,U,heat])

    def __call__(self,graph,path):
        return float(np.dot(self.metrics(graph,path)/self.scales,self.weights))


def mst_lower_bound(graph,remaining,current,end):
    """Undirected relaxation of a directed graph is an admissible length bound.

    It never adds a geometric edge to the actual search graph. Any disconnected
    relaxation proves that the current sparse-graph branch cannot finish.
    """
    remaining=set(remaining)-{current,end}
    if not remaining:
        return graph.segments[(current,end)].length if (current,end) in graph.lookup else math.inf
    def weight(i,j):
        if (i,j) in graph.lookup or (j,i) in graph.lookup:
            return float(np.linalg.norm(graph.points[i]-graph.points[j]))
        return math.inf
    used={min(remaining)}; todo=remaining-used; total=0.
    while todo:
        w,j=min((weight(i,j),j) for i in used for j in todo)
        if not math.isfinite(w): return math.inf
        total+=w; used.add(j); todo.remove(j)
    first=min([graph.segments[(current,j)].length for j in remaining if (current,j) in graph.lookup] or [math.inf])
    last=min([graph.segments[(j,end)].length for j in remaining if (j,end) in graph.lookup] or [math.inf])
    return total+first+last
