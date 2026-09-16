"""PA-LCCS quintic local replacement with explicit unsmoothed-corner status."""
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import LineString, MultiPoint, Point
from .geometry import unit, angle, allowed_intersection


# Numerical coincidence tolerance used only to remove genuinely duplicated
# adjacent coordinates.  The manuscript epsilon_L is a minimum segment-length
# threshold for corner screening and must not be reused as a deduplication
# tolerance.
_DUPLICATE_TOL = 1e-12


def deduplicate_adjacent(points,tol=_DUPLICATE_TOL):
    """Remove only numerically coincident adjacent points and preserve endpoints.

    Short but distinct segments are retained.  If the final coordinate is a
    duplicate of the preceding retained coordinate, the stored coordinate is
    replaced by the original final point so the path endpoint is preserved.
    """
    pts=np.asarray(points,dtype=float)
    if len(pts)<=1:
        return pts.copy()
    out=[pts[0].copy()]
    for p in pts[1:-1]:
        if np.linalg.norm(p-out[-1])>tol:
            out.append(p.copy())
    last=pts[-1].copy()
    if np.linalg.norm(last-out[-1])>tol:
        out.append(last)
    else:
        out[-1]=last
    return np.asarray(out,dtype=float)


def combination(n,k):
    return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))


@dataclass
class SmoothConfig:
    bead_width: float = 1.
    epsilon_L: float = 1e-9
    min_window: float = .05
    theta_min: float = .15
    curvature_min: float = .01
    alpha: float = .45
    beta: float = .8
    gamma: float = 1.
    max_curvature: float = 8.
    deviation: float = .3
    min_speed: float = 1e-6
    tolerance: float = .002
    weights: tuple = (.4,.4,.2)


def control_points(a,b,tin,tout,lam_in,lam_out):
    """Eq. (39): b''(0)=b''(1)=0 and tangent directions match the lines."""
    return np.array([a,a+lam_in/5*tin,a+2*lam_in/5*tin,
                     b-2*lam_out/5*tout,b-lam_out/5*tout,b])


def bezier(control,t):
    n=len(control)-1;t=np.asarray(t)
    basis=np.stack([combination(n,k)*t**k*(1-t)**(n-k) for k in range(n+1)],-1)
    return basis@control


def derivatives(control,t):
    first=5*np.diff(control,axis=0);second=4*np.diff(first,axis=0)
    return bezier(first,t),bezier(second,t)


def split_control(control):
    rows=[control]
    while len(rows[-1])>1: rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    return np.array([row[0] for row in rows]),np.array([row[-1] for row in rows[::-1]])


def adaptive_sample(control,tolerance,depth=0):
    chord=LineString([control[0],control[-1]])
    flat=max(Point(p).distance(chord) for p in control)
    if flat<=tolerance: return np.array([control[0],control[-1]])
    if depth>=20:
        raise ValueError('Bezier subdivision could not establish requested tolerance')
    a,b=split_control(control)
    return np.vstack([adaptive_sample(a,tolerance,depth+1)[:-1],adaptive_sample(b,tolerance,depth+1)])


def curvature_certificate(control,max_curvature,min_speed,depth=0,scale=1.):
    """Conservative Bernstein-hull regularity/curvature certificate.

    Subdivision bounds the full curve, not just sampled maxima. A depth-limited
    uncertain result is rejected. Speed is transformed to the original u scale.
    """
    first=5*np.diff(control,axis=0);second=4*np.diff(first,axis=0)
    speed_lower=MultiPoint(first).convex_hull.distance(Point(0,0))
    cross=[]
    for k in range(8):
        value=0.
        for i in range(5):
            j=k-i
            if 0<=j<4:
                value+=combination(4,i)*combination(3,j)/combination(7,k)*float(first[i,0]*second[j,1]-first[i,1]*second[j,0])
        cross.append(value)
    if speed_lower*scale>=min_speed and max(abs(v) for v in cross)<=max_curvature*speed_lower**3:
        return True
    if depth>=12: return False
    a,b=split_control(control)
    return curvature_certificate(a,max_curvature,min_speed,depth+1,scale*2) and curvature_certificate(b,max_curvature,min_speed,depth+1,scale*2)


def densify(line,step):
    n=max(2,int(math.ceil(line.length/step))+1)
    return [line.interpolate(float(t)) for t in np.linspace(0,line.length,n)]


def deviation_upper(sampled,original,flatness,step):
    """Lipschitz sampling upper bound on bidirectional geometric deviation."""
    a=max(p.distance(original) for p in densify(sampled,step))
    b=max(p.distance(sampled) for p in densify(original,step))
    return max(a,b)+step/2+flatness


def curve_clear_of(control,obstacle,allowed=(),depth=0):
    """Sufficient hull-subdivision separation from a retained geometric set.

    Contacts are permitted only within the declared 1e-8 mm endpoint tolerance.
    Inconclusive intersections are rejected instead of accepted by sampling.
    """
    hull=MultiPoint(control).convex_hull
    if allowed_intersection(hull,obstacle,allowed): return True
    if depth>=14: return False
    a,b=split_control(control)
    return curve_clear_of(a,obstacle,allowed,depth+1) and curve_clear_of(b,obstacle,allowed,depth+1)


def monotonic_projection(control):
    """A strictly positive derivative projection proves no self-intersection."""
    direction=unit(control[-1]-control[0])
    return bool(np.min(5*np.diff(control,axis=0)@direction)>1e-10)


def smooth_path(points,safe,config=None):
    cfg=config or SmoothConfig()
    if min(cfg.bead_width,cfg.deviation,cfg.tolerance,cfg.max_curvature,cfg.min_speed)<=0 or cfg.epsilon_L<0:
        raise ValueError('Positive smoothing limits and nonnegative epsilon_L required')
    if min(cfg.weights)<0 or not np.isclose(sum(cfg.weights),1): raise ValueError('Smoothing weights must sum to one')
    points=deduplicate_adjacent(points)
    original=LineString(points)
    if not original.is_simple or not safe.buffer(1e-8).covers(original):
        return dict(status='infeasible',points=points,corners=[],curves=[])
    candidate=[]
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)]
    for i in range(1,len(points)-1):
        turn=angle(points[i]-points[i-1],points[i+1]-points[i])
        if min(lengths[i-1],lengths[i])<=cfg.epsilon_L: continue
        curvature=2*math.sin(turn/2)/max((lengths[i-1]+lengths[i])/2,1e-12)
        if turn>=cfg.theta_min and curvature>=cfg.curvature_min: candidate.append((i,turn))
    out=[points[0]];records=[];curves=[]
    index_set=[i for i,_ in candidate]
    for i in range(1,len(points)-1):
        if i not in index_set:
            out.append(points[i]);continue
        turn=dict(candidate)[i]
        tin=unit(points[i]-points[i-1]);tout=unit(points[i+1]-points[i])
        clearance=Point(points[i]).distance(safe.boundary)
        pos=index_set.index(i)
        adj_before=cumulative[i]-cumulative[index_set[pos-1]] if pos>0 else math.inf
        adj_after=cumulative[index_set[pos+1]]-cumulative[i] if pos+1<len(index_set) else math.inf
        caps=np.minimum([cfg.alpha*lengths[i-1],cfg.alpha*lengths[i]],
                        np.minimum([cfg.beta*clearance]*2,[adj_before/2,adj_after/2]))
        if min(caps)<cfg.min_window:
            out.append(points[i]);records.append(dict(index=i,status='retained',reason='window_too_small'));continue
        request=max(cfg.min_window,cfg.gamma*cfg.bead_width/2*math.tan(turn/2))
        sm,sp=np.minimum(request,caps)
        a=points[i]-sm*tin;b=points[i]+sp*tout
        local=LineString([a,points[i],b]);length=sm+sp
        t=np.linspace(0,1,101)
        def evaluate(lam,final=False):
            ctrl=control_points(a,b,tin,tout,*lam)
            sample=adaptive_sample(ctrl,cfg.tolerance) if final else bezier(ctrl,t)
            line=LineString(sample)
            first,second=derivatives(ctrl,t)
            speed=np.linalg.norm(first,axis=1)
            curvature=np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])/np.maximum(speed,1e-12)**3
            deviation=line.hausdorff_distance(local)
            delta=line.buffer(cfg.bead_width/2).symmetric_difference(local.buffer(cfg.bead_width/2)).area
            energy=length*np.trapezoid(curvature**2*speed,t)
            obj=cfg.weights[0]*energy+cfg.weights[1]*(deviation/cfg.deviation)**2+cfg.weights[2]*delta/(cfg.bead_width*length)
            constraints=np.array([cfg.max_curvature-curvature.max(),speed.min()-cfg.min_speed,cfg.deviation-deviation])
            return obj,constraints,ctrl,sample
        starts=[np.array([sm,sp])*factor for factor in (1.,1.7,2.3)]
        accepted=[]
        for start in starts:
            res=minimize(lambda x:evaluate(x)[0],start,method='SLSQP',
                         bounds=[(1e-5,2.5*sm),(1e-5,2.5*sp)],
                         constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[1])],
                         options=dict(maxiter=60,ftol=1e-7))
            obj,ineq,ctrl,sample=evaluate(res.x,True)
            if min(ineq)<-1e-7: continue
            # The convex-hull inclusion is sufficient for continuous domain safety.
            if not safe.buffer(1e-8).covers(MultiPoint(ctrl).convex_hull): continue
            if not curvature_certificate(ctrl,cfg.max_curvature,cfg.min_speed): continue
            if not monotonic_projection(ctrl): continue
            before=LineString(np.vstack([np.asarray(out),a]))
            after=LineString(np.vstack([b,points[i+1:]]))
            if not curve_clear_of(ctrl,before,[a]) or not curve_clear_of(ctrl,after,[b]): continue
            # Previously accepted curves are represented by their analytic
            # control hulls. This conservative check protects between samples.
            if any(not curve_clear_of(ctrl,MultiPoint(old).convex_hull) for old in curves): continue
            upper=deviation_upper(LineString(sample),local,cfg.tolerance,cfg.tolerance*2)
            if upper>cfg.deviation: continue
            trial=deduplicate_adjacent(np.vstack([np.asarray(out),sample,points[i+1:]]))
            if not LineString(trial).is_simple: continue
            accepted.append((obj,ctrl,sample,upper))
        if not accepted:
            out.append(points[i]);records.append(dict(index=i,status='retained',reason='no_certified_candidate'));continue
        obj,ctrl,sample,upper=min(accepted,key=lambda v:v[0])
        out.extend(sample)
        curves.append(ctrl.tolist())
        records.append(dict(index=i,status='smoothed',objective=float(obj),deviation_upper=float(upper)))
    out.append(points[-1]);out=deduplicate_adjacent(np.asarray(out))
    if not LineString(out).is_simple or not safe.buffer(1e-8).covers(LineString(out)):
        return dict(status='fallback_original',points=points,corners=records,curves=[])
    status='smoothed' if records and all(r['status']=='smoothed' for r in records) else 'partial_or_unchanged'
    return dict(status=status,points=out,corners=records,curves=curves,
                intersection_check='analytic convex-hull separation, monotonic projection and curvature certificates; 1e-8 mm endpoint tolerance')
