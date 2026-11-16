"""Per-trial outlier removal on (N, 5) point arrays: x, y, z, intensity/Doppler, frame index."""
import functools
import numpy as np
from sklearn.cluster import DBSCAN


def dbscan(P, eps=0.3, ms=4):
    """DBSCAN on (x, y, z) over all frames of a trial; keep the largest cluster."""
    if len(P) < ms:
        return P
    lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(P[:, :3])
    if (lab >= 0).sum() == 0:
        return P
    return P[lab == np.bincount(lab[lab >= 0]).argmax()]


def ror(P, r=0.3, m=4):
    """Radius outlier removal: keep points with >= m-1 neighbours within r (no clustering)."""
    if len(P) < m:
        return P
    from scipy.spatial import cKDTree
    c = np.array([len(x) for x in cKDTree(P[:, :3]).query_ball_point(P[:, :3], r)])
    k = c >= m
    return P[k] if k.any() else P


def gridcc(P, cell=0.15):
    """Largest 26-connected component of occupied grid cells."""
    if len(P) < 4:
        return P
    from scipy import ndimage
    q = np.floor((P[:, :3] - P[:, :3].min(0)) / cell).astype(int)
    dims = q.max(0) + 1
    if np.prod(dims) > 2_000_000:
        return dbscan(P)
    occ = np.zeros(dims, bool)
    occ[q[:, 0], q[:, 1], q[:, 2]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    pl = lab[q[:, 0], q[:, 1], q[:, 2]]
    return P[pl == np.bincount(pl, minlength=n + 1)[1:].argmax() + 1]


def gate(P, r0=0.25):
    """Near-field range gate: drop points closer than r0 to the radar (no clustering)."""
    k = np.linalg.norm(P[:, :3], axis=1) >= r0
    return P[k] if k.sum() >= 1 else P


def gatedb(P, r0=0.25):
    """Near-field gate, then DBSCAN largest cluster."""
    return dbscan(gate(P, r0))


DENOISERS = {'dbscan': dbscan, 'ror': ror, 'grid': gridcc, 'gate': gate, 'gatedb': gatedb}
for _e in (0.15, 0.3, 0.5, 1.0):
    for _m in (3, 4, 6):
        DENOISERS[f'db_{_e}_{_m}'] = functools.partial(dbscan, eps=_e, ms=_m)


def largest_cluster_is_near_field(P, r0=0.25):
    """True if the DBSCAN largest cluster of a trial lies at the radar (median range < r0)."""
    if len(P) < 4:
        return False
    lab = DBSCAN(eps=0.3, min_samples=4).fit_predict(P[:, :3])
    if (lab >= 0).sum() == 0:
        return False
    k = np.bincount(lab[lab >= 0]).argmax()
    return bool(np.median(np.linalg.norm(P[lab == k, :3], axis=1)) < r0)
