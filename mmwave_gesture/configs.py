"""Model and input configurations used in the paper (name -> dict).

kind: 'vox' (TVoxNet), 'radhar' or 'pts'; den: outlier removal ('none', 'dbscan', 'gate', 'gatedb', ...);
norm (voxels) / pnorm (points): 'axis' or 'sample' = per-sample normalisation, 'fixed' = one frame per training split.
Suffixes: _noden raw points, _gate near-field gate only, _gatedb gate then DBSCAN, _fix fixed-scale normalisation.
"""
from .preprocess import DENOISERS

A = dict(kind='vox', n=16, T=8, c=(16, 32, 64), d=128, tmp='conv', aug=True, norm='axis', conv='dws', early=False, den='dbscan')


def va(**k):
    r = dict(A)
    r.update(k)
    return r


M = {
    'tvoxA': A,
    'tvoxF': dict(A, c=(32, 64, 128), d=192),
    # TVoxNet ablations
    'abl_gru': va(tmp='gru'), 'abl_avg': va(tmp='avg'), 'abl_T1': va(T=1), 'abl_T4': va(T=4), 'abl_T12': va(T=12),
    'abl_std3d': va(conv='std'), 'abl_early': va(early=True), 'abl_iso': va(norm='iso'),
    'abl_n12': va(n=12), 'abl_n20': va(n=20), 'abl_n24': va(n=24),
    'abl_noden': va(den='none'), 'abl_ror': va(den='ror'), 'abl_grid': va(den='grid'), 'abl_int': va(cint=True),
    'abl_noaug': va(aug=False),
    # baselines
    'radhar': dict(kind='radhar', n=16, T=8, aug=True, norm='axis', den='dbscan'),
    'pointnet_t_aug': dict(kind='pts', net='pointnet', aug=True, den='dbscan'),
    'dgcnn_t_aug': dict(kind='pts', net='dgcnn', aug=True, den='dbscan'),
    'pn2_t_aug': dict(kind='pts', net='pn2', aug=True, den='dbscan'),
    'p4t_aug': dict(kind='pts', net='p4t', aug=True, den='dbscan'),
    'pantomime_aug': dict(kind='pts', net='pantomime', aug=True, den='dbscan'),
}
# compute-matched (~10 MMACs) point networks
M['pointnet_t_tiny'] = dict(M['pointnet_t_aug'], kw=dict(w=(48, 96, 320), h=(256, 128)))
M['p4t_tiny'] = dict(M['p4t_aug'], kw=dict(S=64, d=64, e=(32, 64)))
# DBSCAN eps x minPts sweep for TVoxNet-A
for _k in [k for k in DENOISERS if k.startswith('db_')]:
    M['tvoxA_' + _k] = va(den=_k)

BASELINES = ['pointnet_t_aug', 'dgcnn_t_aug', 'pn2_t_aug', 'p4t_aug', 'pantomime_aug', 'radhar', 'pointnet_t_tiny', 'p4t_tiny']
for _b in BASELINES:
    vox = M[_b]['kind'] == 'radhar'
    fixed = dict(norm='fixed') if vox else dict(pnorm='fixed')
    M[_b + '_noden'] = dict(M[_b], den='none')
    M[_b + '_gate'] = dict(M[_b], den='gate')
    M[_b + '_gatedb'] = dict(M[_b], den='gatedb')
    M[_b + '_fix'] = dict(M[_b], **fixed)
    M[_b + '_noden_fix'] = dict(M[_b], den='none', **fixed)
    M[_b + '_gate_fix'] = dict(M[_b], den='gate', **fixed)
    M[_b + '_gatedb_fix'] = dict(M[_b], den='gatedb', **fixed)
# TVoxNet-A input variants (raw = abl_noden)
M['abl_gate'] = va(den='gate')
M['abl_gatedb'] = va(den='gatedb')
M['tvoxA_fix'] = va(norm='fixed')
M['abl_noden_fix'] = va(norm='fixed', den='none')
M['abl_gate_fix'] = va(den='gate', norm='fixed')
M['abl_gatedb_fix'] = va(den='gatedb', norm='fixed')
