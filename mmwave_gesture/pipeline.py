"""Batched GPU/CPU input pipeline: augmentation, temporal voxelisation and point resampling."""
import math
import torch

NPTS = 256


class PackedData:
    """All points of a dataset part on one device; batch() gathers the points of selected trials."""

    def __init__(self, pts, off, device):
        P = torch.tensor(pts, dtype=torch.float32)
        P[:, 3] = torch.nan_to_num(P[:, 3])
        self.P = P.to(device)
        self.off = torch.tensor(off, device=device)

    def batch(self, b):
        dev = self.P.device
        st = self.off[b]
        L = self.off[b + 1] - st
        cs = torch.cumsum(L, 0) - L
        bid = torch.repeat_interleave(torch.arange(len(b), device=dev), L)
        idx = torch.repeat_interleave(st, L) + torch.arange(int(L.sum()), device=dev) - torch.repeat_interleave(cs, L)
        return self.P[idx], bid


def seg_min(x, bid, B):
    return torch.full((B, x.shape[1]), float('inf'), device=x.device).scatter_reduce(0, bid[:, None].expand_as(x), x, 'amin')


def seg_max(x, bid, B):
    return torch.full((B, x.shape[1]), float('-inf'), device=x.device).scatter_reduce(0, bid[:, None].expand_as(x), x, 'amax')


def augment(P, bid, B, aug, g, want_c3=False):
    """Training-time augmentation (point dropout, rotation, scaling, jitter, temporal crop and warp); returns xyz, t in [0, 1], bid."""
    dev = P.device
    xyz = P[:, :3].clone()
    fr = P[:, 4:5].clone()
    c3 = P[:, 3].clone()
    if aug:
        keep = torch.rand(len(P), device=dev, generator=g) > torch.rand(B, device=dev, generator=g)[bid] * 0.2
        first = torch.ones(len(bid), dtype=torch.bool, device=dev)
        first[1:] = bid[1:] != bid[:-1]
        keep |= first  # never empty a sample
        xyz, fr, bid, c3 = xyz[keep], fr[keep], bid[keep], c3[keep]
        cnt = torch.zeros(B, device=dev).index_add_(0, bid, torch.ones(len(bid), device=dev)).clamp(min=1)
        cen = torch.zeros(B, 3, device=dev).index_add_(0, bid, xyz) / cnt[:, None]
        th = (torch.rand(B, device=dev, generator=g) * 2 - 1) * math.radians(15)
        c, s_ = torch.cos(th)[bid], torch.sin(th)[bid]
        q = xyz - cen[bid]
        q = torch.stack([c * q[:, 0] - s_ * q[:, 1], s_ * q[:, 0] + c * q[:, 1], q[:, 2]], 1)
        q = q * (0.9 + 0.2 * torch.rand(B, 3, device=dev, generator=g))[bid]
        xyz = q + cen[bid] + 0.01 * torch.randn(q.shape, device=dev, generator=g)
    lo = seg_min(fr, bid, B)
    hi = seg_max(fr, bid, B)
    tn = ((fr - lo[bid]) / (hi - lo).clamp(min=1e-9)[bid])[:, 0]
    if aug:
        a = torch.rand(B, device=dev, generator=g) * 0.1
        e = 1 - torch.rand(B, device=dev, generator=g) * 0.1
        gam = torch.exp((torch.rand(B, device=dev, generator=g) * 2 - 1) * 0.2)
        tn = ((tn - a[bid]) / (e - a)[bid]).clamp(0, 1) ** gam[bid]
    if want_c3:
        return xyz, tn, bid, c3
    return xyz, tn, bid


def voxelize(P, bid, B, n, T, norm='axis', aug=False, g=None, cint=False, fix=None):
    """T time slices of n^3 log(1+count) grids.

    norm='axis': per-sample, per-axis min-max; 'iso': per-sample isotropic box;
    'fixed': one box (lo, hi) = fix from the training split for every sample.
    cint=True adds a second channel with the mean column-3 value per voxel.
    """
    dev = P.device
    xyz, tn, bid, c3 = augment(P, bid, B, aug, g, True)
    mn = seg_min(xyz, bid, B)
    mx = seg_max(xyz, bid, B)
    ext = mx - mn
    if norm == 'fixed':
        lo, hi = fix
        mn = lo.expand(B, 3)
        ext = (hi - lo).expand(B, 3)
    if norm == 'iso':
        m = ext.max(1, keepdim=True)[0]
        mn = mn - (m - ext) / 2
        ext = m.expand_as(ext)
    ii = ((xyz - mn[bid]) / ext.clamp(min=1e-9)[bid] * n).floor().long().clamp(0, n - 1)
    ti = (tn * T).floor().long().clamp(0, T - 1)
    flat = (((bid * T + ti) * n + ii[:, 0]) * n + ii[:, 1]) * n + ii[:, 2]
    v = torch.zeros(B * T * n ** 3, device=dev).index_add_(0, flat, torch.ones(len(flat), device=dev))
    if not cint:
        return torch.log1p(v.view(B, T, n, n, n))
    w = torch.zeros(B * T * n ** 3, device=dev).index_add_(0, flat, c3) / v.clamp(min=1)
    return torch.stack([torch.log1p(v).view(B, T, n, n, n), torch.log1p(w.clamp(min=0)).view(B, T, n, n, n)], 2)


def pointify(P, bid, B, aug=False, g=None, gs=None, pnorm='sample', fixp=None):
    """NPTS points per sample (sampled with replacement): normalised xyz + t in [-1, 1].

    pnorm='sample': per-sample z-score of xyz; 'fixed': one per-axis (mean, std) = fixp from the training split.
    """
    dev = P.device
    xyz, tn, bid = augment(P, bid, B, aug, g)
    L = torch.bincount(bid, minlength=B)
    off = torch.cumsum(L, 0) - L
    Lf = L.clamp(min=1).float()[:, None]
    if pnorm == 'fixed':
        mu, sd = fixp
        z = (xyz - mu) / sd
    else:
        mu = torch.zeros(B, 3, device=dev).index_add_(0, bid, xyz) / Lf
        var = torch.zeros(B, 3, device=dev).index_add_(0, bid, (xyz - mu[bid]) ** 2) / Lf
        sd = var.sqrt()
        sd = torch.where(sd == 0, torch.ones_like(sd), sd)
        z = (xyz - mu[bid]) / sd[bid]
    u = torch.rand(B, NPTS, device=dev, generator=gs if gs is not None else g)
    idx = off[:, None] + (u * L[:, None]).long().clamp(max=(L - 1).clamp(min=0)[:, None])
    return torch.cat([z[idx], (tn[idx] * 2 - 1)[..., None]], -1)
