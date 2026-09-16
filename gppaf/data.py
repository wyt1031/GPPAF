"""Measured-data grouping and training-only augmentation for TC-DSFNet."""
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def grouped_split(groups, fold=0, folds=5, seed=2026):
    groups=np.asarray(groups)
    if not 0 <= fold < folds:
        raise ValueError('Invalid outer fold')
    tr,te=list(GroupKFold(folds).split(np.zeros(len(groups)),groups=groups))[fold]
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=groups[tr]))
    result=dict(train=tr[a],validation=tr[b],test=te)
    for u,v in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(groups[result[u]]) & set(groups[result[v]])
    return result


def augment_training(x,y,multiplier=5,seed=0):
    """Nearest-neighbor convex interpolation within a supplied training fold.

    Parameters are normalized by their training spread for neighbor selection.
    Interpolated records have weight 0.3; measured rows retain weight 1.
    Parent indices allow an independent leakage audit.
    """
    rng=np.random.RandomState(seed)
    n=len(x)
    if n<2 or multiplier<1:
        raise ValueError('At least two training rows and multiplier >= 1 required')
    z=(x-x.mean(0))/np.maximum(x.std(0),1e-8)
    d=((z[:,None,:]-z[None,:,:])**2).sum(-1)
    np.fill_diagonal(d,np.inf)
    neighbors=np.argsort(d,axis=1)[:,:min(5,n-1)]
    i=rng.randint(n,size=n*(multiplier-1))
    j=neighbors[i,rng.randint(neighbors.shape[1],size=len(i))]
    alpha=rng.uniform(size=(len(i),1))
    return (np.concatenate([x,alpha*x[i]+(1-alpha)*x[j]]),
            np.concatenate([y,alpha*y[i]+(1-alpha)*y[j]]),
            np.concatenate([np.ones(n),np.full(len(i),.3)]),np.column_stack([i,j]))


def bootstrap_signs(x,y,groups=None,repeats=1000,stability=.95,seed=0):
    """Group bootstrap of standardized linear coefficients, training rows only."""
    rng=np.random.RandomState(seed)
    groups=np.arange(len(x)) if groups is None else np.asarray(groups)
    unique=np.unique(groups)
    z=(x-x.mean(0))/np.maximum(x.std(0),1e-8)
    z=np.column_stack([np.ones(len(z)),z])
    coefficients=[]
    for _ in range(repeats):
        idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        coefficients.append(np.linalg.lstsq(z[idx],y[idx],rcond=None)[0][1:])
    coeff=np.asarray(coefficients)
    positive=(coeff>1e-10).mean(0)
    negative=(coeff < -1e-10).mean(0)
    signs=[]
    for k in range(x.shape[1]):
        for r in range(y.shape[1]):
            if positive[k,r]>=stability: signs.append((k,r,1))
            elif negative[k,r]>=stability: signs.append((k,r,-1))
    return signs


def layer_trends(y,repeats=1000,stability=.95,seed=0):
    """Screen increasing width/decreasing height using condition bootstrap."""
    rng=np.random.RandomState(seed)
    t=np.arange(y.shape[1]); t=t-t.mean()
    slopes=np.einsum('n,bnr->br',t,y)/(t*t).sum()
    samples=np.array([slopes[rng.randint(len(y),size=len(y))].mean(0) for _ in range(repeats)])
    return (bool((samples[:,0]>0).mean()>=stability), bool((samples[:,1]<0).mean()>=stability))
