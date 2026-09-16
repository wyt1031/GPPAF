"""Monotonic rational-quadratic splines with analytic inverse and Jacobian.

Used by TC-DSFNet Eq. (3). Outside [-bound, bound] the map is identity.
The bin formula follows Durkan et al., Neural Spline Flows (2019).
This is an independent implementation, not a copy of a third-party package.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def rational_quadratic(x, raw, inverse=False, bound=5., minimum=1e-3):
    """Transform [..., D] values using [..., D, 3*K-1] parameters.

    The returned log determinant is elementwise. End derivatives are fixed at
    one to make the linear tails continuously differentiable.
    """
    bins = (raw.shape[-1] + 1) // 3
    if bins * minimum >= 1 or raw.shape[-1] != 3*bins-1:
        raise ValueError('Invalid spline bin parameterization')
    widths = minimum + (1-bins*minimum)*F.softmax(raw[..., :bins], -1)
    heights = minimum + (1-bins*minimum)*F.softmax(raw[..., bins:2*bins], -1)
    widths, heights = widths*(2*bound), heights*(2*bound)
    zeros = torch.zeros_like(widths[..., :1])
    xknots = torch.cat([zeros, widths.cumsum(-1)], -1)-bound
    yknots = torch.cat([zeros, heights.cumsum(-1)], -1)-bound
    interior = minimum + F.softplus(raw[..., 2*bins:])
    derivatives = torch.cat([torch.ones_like(zeros), interior, torch.ones_like(zeros)], -1)
    inside = (x >= -bound) & (x <= bound)
    safe = x.clamp(-bound, bound)
    knots = yknots if inverse else xknots
    idx = (safe.unsqueeze(-1) >= knots[..., 1:]).sum(-1).clamp(max=bins-1)
    def take(a, offset=0):
        return a.gather(-1, (idx+offset).unsqueeze(-1)).squeeze(-1)
    x0, y0, w, h = take(xknots), take(yknots), take(widths), take(heights)
    d0, d1 = take(derivatives), take(derivatives, 1)
    slope = h/w
    if inverse:
        ydiff = safe-y0
        a = ydiff*(d0+d1-2*slope)+h*(slope-d0)
        b = h*d0-ydiff*(d0+d1-2*slope)
        c = -slope*ydiff
        disc = (b*b-4*a*c).clamp_min(0)
        # Stable root in [0,1]; the linear case is included by this expression.
        theta = (2*c)/(-b-torch.sqrt(disc)).clamp(max=-1e-15)
        theta = theta.clamp(0, 1)
        y = x0+theta*w
    else:
        theta = (safe-x0)/w
        numerator = h*(slope*theta**2+d0*theta*(1-theta))
        denom = slope+(d0+d1-2*slope)*theta*(1-theta)
        y = y0+numerator/denom
    denom = slope+(d0+d1-2*slope)*theta*(1-theta)
    derivative = slope**2*(d1*theta**2+2*slope*theta*(1-theta)+d0*(1-theta)**2)/denom**2
    logdet = torch.log(derivative.clamp_min(torch.finfo(x.dtype).tiny))
    return torch.where(inside, y, x), torch.where(inside, -logdet if inverse else logdet, torch.zeros_like(x))


class SplineCoupling(nn.Module):
    def __init__(self, dimension, context, hidden=64, bins=8, parity=0):
        super().__init__()
        fixed = [i for i in range(dimension) if i % 2 == parity]
        moved = [i for i in range(dimension) if i not in fixed]
        self.register_buffer('fixed', torch.tensor(fixed, dtype=torch.long))
        self.register_buffer('moved', torch.tensor(moved, dtype=torch.long))
        self.bins = bins
        self.net = nn.Sequential(nn.Linear(len(fixed)+context, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, len(moved)*(3*bins-1)))
        nn.init.normal_(self.net[-1].weight,mean=0.,std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias.view(len(moved), -1)[:, 2*bins:] = math.log(math.expm1(1-1e-3))

    def forward(self, x, context, inverse=False):
        raw = self.net(torch.cat([x[..., self.fixed], context], -1))
        raw = raw.view(*x.shape[:-1], len(self.moved), 3*self.bins-1)
        changed, logdet = rational_quadratic(x[..., self.moved], raw, inverse)
        y = x.clone()
        y[..., self.moved] = changed
        return y, logdet.sum(-1)
