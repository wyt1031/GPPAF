"""Raw-array performance metrics; no manuscript figure values are embedded."""
import numpy as np
from scipy.stats import wilcoxon,rankdata


def paired_comparison(reference,baselines,seed=0,resamples=2000):
    """Paired bootstrap CI and Wilcoxon tests with Holm correction.

    Arrays must include a predeclared finite failure penalty where runs failed.
    Success rate is reported separately by the benchmark driver.
    """
    ref=np.asarray(reference,float);rng=np.random.RandomState(seed);result={}
    pvalues=[]
    for name,values in baselines.items():
        values=np.asarray(values,float)
        if values.shape!=ref.shape or not np.isfinite(values).all() or not np.isfinite(ref).all():
            raise ValueError('Finite aligned paired values required; do not drop failures')
        diff=values-ref
        nonzero=diff[np.abs(diff)>1e-12]
        if len(nonzero)==0:
            p=1.;method='all differences zero'
        elif len(nonzero)<=20:
            # Exact conditional sign-flip distribution also handles tied ranks.
            ranks=rankdata(np.abs(nonzero))
            distribution=np.array([0.])
            for rank in ranks: distribution=np.r_[distribution+rank,distribution-rank]
            observed=abs(float(np.dot(ranks,np.sign(nonzero))))
            p=float((np.abs(distribution)>=observed-1e-10).mean())
            method='exact conditional Wilcoxon sign-flip enumeration'
        else:
            p=float(wilcoxon(diff,alternative='two-sided').pvalue)
            method='scipy Wilcoxon'
        bootstrap=np.array([diff[rng.randint(len(diff),size=len(diff))].mean() for _ in range(resamples)])
        result[name]=dict(mean_advantage=float(diff.mean()),advantage_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                          paired_dominance=float((diff>0).mean()+.5*(diff==0).mean()),p_raw=p,p_method=method)
        pvalues.append((p,name))
    last=0.
    for i,(p,name) in enumerate(sorted(pvalues)):
        last=max(last,min(1.,(len(pvalues)-i)*p));result[name]['p_holm']=last
    return result


def stress_metrics(fields,area_weights=None,threshold=.45):
    arrays=[np.asarray(a,float) for a in fields]
    if any(a.size==0 or np.any(a<0) or not np.isfinite(a).all() for a in arrays): raise ValueError('Finite nonnegative stress fields required')
    maximum=max(float(a.max()) for a in arrays)
    if maximum<=0: raise ValueError('Global stress maximum must be positive')
    result=[]
    for i,a in enumerate(arrays):
        w=np.ones_like(a) if area_weights is None else np.asarray(area_weights[i],float)
        if w.shape!=a.shape or min(w)<0 or w.sum()<=0: raise ValueError('Invalid area weights')
        z=a/maximum;mean=float(np.average(z,weights=w))
        variance=float(np.average((z-mean)**2,weights=w))
        result.append(dict(high_stress_fraction=float(np.average(z>=threshold,weights=w)),
                           coefficient_of_variation=float(np.sqrt(variance)/mean) if mean>0 else None,
                           normalized_mean=mean,area_weighted=area_weights is not None))
    return result


def point_cloud_metrics(clouds,boundary_masks=None,area_weights=None,threshold=.18):
    arrays=[np.asarray(a,float) for a in clouds]
    if any(a.ndim!=2 or a.shape[1]!=3 or len(a)==0 or not np.isfinite(a).all() for a in arrays): raise ValueError('Finite nonempty Nx3 point clouds required')
    low=min(a[:,2].min() for a in arrays);high=max(a[:,2].max() for a in arrays)
    if high<=low: raise ValueError('Global height range is zero')
    result=[]
    for i,a in enumerate(arrays):
        z=(a[:,2]-low)/(high-low);residual=np.abs(z-np.median(z))
        w=np.ones(len(z)) if area_weights is None else np.asarray(area_weights[i],float)
        if w.shape!=z.shape or min(w)<0 or w.sum()<=0: raise ValueError('Invalid point area weights')
        item=dict(normalized_height_range90_10=float(np.quantile(z,.9)-np.quantile(z,.1)),
                  median_absolute_residual=float(np.median(residual)),
                  outlier_fraction=float(np.average(residual>threshold,weights=w)),area_weighted=area_weights is not None)
        if boundary_masks is not None:
            b=np.asarray(boundary_masks[i],bool)
            if b.shape!=z.shape: raise ValueError('Invalid boundary mask')
            item.update(boundary_median=float(np.median(residual[b])) if b.any() else None,
                        interior_median=float(np.median(residual[~b])) if (~b).any() else None)
        result.append(item)
    return dict(global_height_min=float(low),global_height_max=float(high),metrics=result)
