"""Bead-centerline geometry and heterogeneous graph, Eqs. (12)-(18),(26)-(27)."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import unary_union


def unit(v):
    v=np.asarray(v,dtype=float)
    n=np.linalg.norm(v)
    if n<1e-12: raise ValueError('Zero direction vector')
    return v/n


def angle(a,b):
    return float(np.arccos(np.clip(unit(a).dot(unit(b)),-1,1)))


def point_parts(g):
    if g.is_empty: return []
    if g.geom_type=='Point': return [np.array(g.coords[0])]
    if g.geom_type in ('LineString','LinearRing'):
        return [np.array(g.coords[0]),np.array(g.coords[-1])]
    return [p for sub in g.geoms for p in point_parts(sub)]


def allowed_intersection(a,b,allowed=(),tol=1e-8):
    """Only explicitly allowed endpoint contacts may remain in an intersection."""
    hit=a.intersection(b)
    if hit.is_empty: return True
    if not allowed: return False
    # Buffer only the contacts, never the full obstacle or connecting segment.
    permitted=unary_union([Point(p).buffer(tol) for p in allowed])
    return hit.difference(permitted).is_empty


@dataclass
class PathGraph:
    points: np.ndarray
    node_types: np.ndarray
    contour_ids: np.ndarray
    edge_index: np.ndarray
    relations: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    region: object
    safe: object
    bead_width: float
    spacing: float

    def __post_init__(self):
        self.lookup={tuple(e):k for k,e in enumerate(self.edge_index)}
        self.neighbors=[[] for _ in self.points]
        for i,j in self.edge_index: self.neighbors[i].append(int(j))
        self.segments={(int(i),int(j)):LineString([self.points[i],self.points[j]]) for i,j in self.edge_index}


def build_graph(outer,holes=(),bead_width=1.,fill_rate=1.,phase=(0.,0.),
                k_neighbors=12,max_scale=2.1,points=None):
    """Construct offsets and directed K-nearest feasible edges.

    Relation assignment used by GPPAF: B connects points on the same
    offset ring, F connects interior pairs, and C connects the remaining
    feasible region-connection pairs.
    Boundary segments remain straight and must pass the full-domain predicate.
    """
    if bead_width<=0 or not 0<fill_rate<=1 or k_neighbors<1 or max_scale<=0:
        raise ValueError('Invalid process or graph sampling configuration')
    region=Polygon(outer,holes)
    if not region.is_valid or region.area<=0: raise ValueError('Invalid polygon or hole topology')
    safe=region.buffer(-bead_width/2,quad_segs=16,join_style=2)
    if safe.is_empty: raise ValueError('No feasible centerline region after bead-width offset')
    g=bead_width/fill_rate
    parts=[safe] if safe.geom_type=='Polygon' else list(safe.geoms)
    rings=[]
    for part in parts:
        rings.append((LineString(part.exterior),1,len(rings)))
        for r in part.interiors: rings.append((LineString(r),2,len(rings)))
    bounds=region.bounds
    span=np.maximum(np.array(bounds[2:])-bounds[:2],1e-8)
    xs=np.arange(bounds[0]+phase[0]*g,bounds[2]+1e-9,g)
    ys=np.arange(bounds[1]+phase[1]*g,bounds[3]+1e-9,g)
    candidates=[]
    if points is None:
        for x in xs:
            for y in ys:
                p=np.array([x,y])
                if safe.covers(Point(p)): candidates.append(p)
        lines=[LineString([(x,bounds[1]-g),(x,bounds[3]+g)]) for x in xs]
        lines += [LineString([(bounds[0]-g,y),(bounds[2]+g,y)]) for y in ys]
        for ring,_,_ in rings:
            for line in lines: candidates.extend(point_parts(line.intersection(ring)))
    else:
        candidates=list(np.asarray(points,dtype=float))
    unique={tuple(np.round(p,9)):np.asarray(p,dtype=float) for p in candidates}
    p=np.array(sorted(unique.values(),key=lambda a:(a[1],a[0])))
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2: raise ValueError('At least two distinct points required')
    types=[]; ids=[]
    for xy in p:
        if not safe.buffer(1e-8).covers(Point(xy)): raise ValueError('Input point outside centerline domain')
        matches=[(ring.distance(Point(xy)),t,i) for ring,t,i in rings]
        distance,t,i=min(matches)
        types.append(t if distance<1e-7 else 0); ids.append(i if distance<1e-7 else -1)
    types=np.array(types); ids=np.array(ids)
    distances=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    outerline=LineString(region.exterior)
    holelines=[LineString(r) for r in region.interiors]
    node_features=[]
    for i,xy in enumerate(p):
        # For hole-free regions the hole distance is the bounding-box diagonal.
        dh=min([h.distance(Point(xy)) for h in holelines] or [np.linalg.norm(span)])
        density=float((distances[i]<=1.5*g).sum())/8.
        node_features.append([*((xy-bounds[:2])/span),outerline.distance(Point(xy))/g,dh/g,density,*np.eye(3)[types[i]]])
    edges=[]; rel=[]; ef=[]
    safe_tol=safe.buffer(1e-8)
    for i in range(len(p)):
        for j in np.argsort(distances[i],kind='mergesort')[:k_neighbors]:
            if distances[i,j]>max_scale*g: continue
            line=LineString([p[i],p[j]])
            if not safe_tol.covers(line): continue
            r=1 if ids[i]>=0 and ids[i]==ids[j] else (0 if types[i]==types[j]==0 else 2)
            direction=(p[j]-p[i])/distances[i,j]
            mid=(p[i]+p[j])/2
            edges.append((i,int(j)));rel.append(r)
            ef.append([distances[i,j]/g,*direction,Point(mid).distance(safe.boundary)/g,*np.eye(3)[r]])
    if not edges: raise ValueError('No feasible candidate edges')
    return PathGraph(p,types,ids,np.array(edges),np.array(rel),np.array(node_features),np.array(ef),region,safe,bead_width,g)


def path_feasible(graph,path,start=None,end=None,complete=True):
    if len(path)!=len(set(path)): return False
    if not path or min(path)<0 or max(path)>=len(graph.points): return False
    if complete and set(path)!=set(range(len(graph.points))): return False
    if start is not None and path[0]!=start: return False
    if end is not None and complete and path[-1]!=end: return False
    for i,j in zip(path[:-1],path[1:]):
        if (i,j) not in graph.lookup: return False
    if len(path)<2: return True
    line=LineString(graph.points[path])
    return line.is_simple and graph.safe.buffer(1e-8).covers(line)


def coverage(graph,path):
    if len(path)<2: return 0.
    footprint=LineString(graph.points[path]).buffer(graph.bead_width/2,quad_segs=32)
    return footprint.intersection(graph.region).area/graph.region.area
