"""User-level statistics of the paper from the per-trial predictions written by train.py.

Per user: accuracy averaged over 3 seeds (users with >= 50 test trials). Paired differences are tested with a
two-sided Wilcoxon signed-rank test and a sign-flip permutation test; 95% CIs bootstrap over users; Holm-Bonferroni
correction over all tests of the family. Also prints pooled accuracies (mean over seeds) per model and input.

Usage: python -m mmwave_gesture.stats --data data --runs runs --out stats.json
"""
import argparse, json, os
import numpy as np
from scipy.stats import wilcoxon

from .preprocess import largest_cluster_is_near_field

NAMES = {'pointnet_t_aug': 'PointNet+t', 'pn2_t_aug': 'PointNet++', 'dgcnn_t_aug': 'DGCNN-lite+t', 'p4t_aug': 'P4T-lite',
         'pantomime_aug': 'Pantomime-style', 'radhar': 'RadHAR', 'tvoxA': 'TVoxNet-A', 'pointnet_t_tiny': 'PointNet+t-tiny',
         'p4t_tiny': 'P4T-tiny', 'tvoxF': 'TVoxNet-F'}
LOCAL = ['pn2_t_aug', 'dgcnn_t_aug', 'p4t_aug', 'p4t_tiny', 'pantomime_aug']
GLOBAL = ['pointnet_t_aug', 'pointnet_t_tiny', 'radhar', 'tvoxA']
PANT_PROTOS = ['LUO5', 'open->office', 'office->open']


def variant(m, v):
    """Configuration name of model m with input variant v.

    Per-sample normalisation: DB, raw, gate, gateDB. Fixed scale: fixDB, fixraw, fixgate, fixgateDB.
    """
    if m == 'tvoxA':
        return {'DB': 'tvoxA', 'raw': 'abl_noden', 'gate': 'abl_gate', 'gateDB': 'abl_gatedb', 'fixDB': 'tvoxA_fix',
                'fixraw': 'abl_noden_fix', 'fixgate': 'abl_gate_fix', 'fixgateDB': 'abl_gatedb_fix'}[v]
    return m + {'DB': '', 'raw': '_noden', 'gate': '_gate', 'gateDB': '_gatedb', 'fixDB': '_fix', 'fixraw': '_noden_fix',
                'fixgate': '_gate_fix', 'fixgateDB': '_gatedb_fix'}[v]


def load_preds(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            d[r['key']] = (np.array(r['idx']), np.array(r['pred']))
    return d


class Stats:
    def __init__(self, data, runs, seed=0):
        self.rng = np.random.default_rng(seed)
        P = np.load(os.path.join(data, 'pantomime_primary.npz'))
        H = np.load(os.path.join(data, 'mhomeges_primary.npz'))
        self.ds = {'pant': (load_preds(os.path.join(runs, 'pantomime_preds.jsonl')), P['gesture'], P['user']),
                   'mh': (load_preds(os.path.join(runs, 'mhomeges_preds.jsonl')), H['gesture'], H['user'])}
        self.mask = None
        self.rows = []
        cache = os.path.join(runs, 'near_field_failures.npy')
        if not os.path.exists(cache):
            off, pts = P['off'], P['pts']
            np.save(cache, np.array([largest_cluster_is_near_field(pts[a:b]) for a, b in zip(off[:-1], off[1:])]))
        self.failures = np.load(cache)

    def peruser(self, ds, m, proto):
        preds, y, U = self.ds[ds]
        out = []
        for s in range(3):
            ks = [f'{m}|LUO5|{s}|{k}' for k in range(5)] if proto == 'LUO5' else [f'{m}|{proto}|{s}']
            if not all(k in preds for k in ks):
                continue
            ii = np.concatenate([preds[k][0] for k in ks])
            pp = np.concatenate([preds[k][1] for k in ks])
            if self.mask is not None and ds == 'pant':
                keep = self.mask[ii]
                ii, pp = ii[keep], pp[keep]
            c = pp == y[ii]
            us = U[ii]
            out.append({u: (c[us == u].mean(), (us == u).sum()) for u in np.unique(us)})
        if len(out) < 3:
            return None
        users = [u for u in sorted(out[0]) if out[0][u][1] >= 50]
        return users, np.array([[o[u][0] for u in users] for o in out]).mean(0) * 100

    def boot(self, d, n=20000):
        b = self.rng.choice(d, (n, len(d))).mean(1)
        return [float(x) for x in np.percentile(b, [2.5, 97.5])]

    def signflip(self, d, n=200000):
        obs = abs(d.mean())
        s = self.rng.choice([-1, 1], size=(n, len(d)))
        return (1 + (np.abs((s * d).mean(1)) >= obs - 1e-12).sum()) / (n + 1)

    def _row(self, fam, ds, proto, a, b, label, d):
        self.rows.append(dict(fam=fam, ds=ds, proto=proto, a=a, b=b, label=label, n=len(d), mean=float(d.mean()),
                              median=float(np.median(d)), pos=int((d > 0).sum()), neg=int((d < 0).sum()),
                              p_wil=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, p_perm=float(self.signflip(d)), ci=self.boot(d)))

    def test(self, ds, proto, a, b, label, fam):
        """Per-user accuracy of b minus a."""
        ra, rb = self.peruser(ds, a, proto), self.peruser(ds, b, proto)
        if ra is None or rb is None:
            print('missing:', ds, proto, a, b)
            return
        assert ra[0] == rb[0]
        self._row(fam, ds, proto, a, b, label, rb[1] - ra[1])

    def contrast(self, ds, proto, loc, glo, label, fam):
        """Per user: mean (b - a) over the local pairs minus mean (b - a) over the global pairs."""
        D = {}
        for grp, pairs in [('l', loc), ('g', glo)]:
            diffs = []
            for a, b in pairs:
                ra, rb = self.peruser(ds, a, proto), self.peruser(ds, b, proto)
                if ra and rb:
                    diffs.append(rb[1] - ra[1])
            if not diffs:
                print('missing contrast:', ds, proto, label)
                return
            D[grp] = np.mean(diffs, 0)
        self._row(fam, ds, proto, 'global', 'local', label, D['l'] - D['g'])

    def holm(self):
        for key in ['p_perm', 'p_wil']:
            ps = np.array([r[key] for r in self.rows])
            adj, run = np.empty(len(ps)), 0
            for rank, i in enumerate(np.argsort(ps)):
                run = max(run, min(1, (len(ps) - rank) * ps[i]))
                adj[i] = run
            for r, v in zip(self.rows, adj):
                r[key + '_holm'] = float(v)

    def pooled(self, ds, m, proto):
        preds, y, _ = self.ds[ds]
        v = []
        for s in range(3):
            ks = [f'{m}|LUO5|{s}|{k}' for k in range(5)] if proto == 'LUO5' else [f'{m}|{proto}|{s}']
            if not all(k in preds for k in ks):
                return None
            ii = np.concatenate([preds[k][0] for k in ks])
            pp = np.concatenate([preds[k][1] for k in ks])
            v.append((pp == y[ii]).mean() * 100)
        return float(np.mean(v)), float(np.std(v, ddof=1))


def paper_family(S):
    raw = {m: variant(m, 'raw') for m in NAMES if m != 'tvoxF'}
    loc = [(m, raw[m]) for m in LOCAL]
    glo = [(m, raw[m]) for m in GLOBAL]
    loc_alt, glo_alt = loc[:-1], glo + [loc[-1]]  # sensitivity: Pantomime-style counted as global
    for proto in PANT_PROTOS:
        S.contrast('pant', proto, loc, glo, 'interaction', 'interaction')
        S.contrast('pant', proto, loc_alt, glo_alt, 'interaction (Pantomime-style global)', 'interaction_alt')
        for m in raw:
            S.test('pant', proto, m, raw[m], NAMES[m] + ' raw-DB', 'raw_vs_db')
    S.mask = ~S.failures
    for proto in PANT_PROTOS:
        S.contrast('pant', proto, loc, glo, 'interaction excl. near-field failure trials', 'interaction_mask')
    S.mask = None
    for m in NAMES:
        if m != 'tvoxA':
            S.test('pant', 'LUO5', 'tvoxA', m, NAMES[m] + '-A (DB)', 'compute')
    for a, b, l in [('tvoxA', 'abl_gate', 'A gate-DB'), ('abl_noden', 'abl_gate', 'A gate-raw'), ('p4t_aug', 'p4t_aug_gate', 'P4T gate-DB'),
                    ('p4t_aug_noden', 'p4t_aug_gate', 'P4T gate-raw'), ('pointnet_t_aug', 'pointnet_t_aug_gate', 'PN+t gate-DB'),
                    ('pointnet_t_aug_noden', 'pointnet_t_aug_gate', 'PN+t gate-raw')]:
        for proto in PANT_PROTOS:
            S.test('pant', proto, a, b, l, 'gate')
    S.test('pant', 'LUO5', 'tvoxA', 'abl_iso', 'iso-A (DB)', 'ablation')
    S.test('pant', 'LUO5', 'tvoxA', 'abl_grid', 'gridCC-A', 'ablation')
    for proto in ['LUO5', 'near->far']:
        S.contrast('mh', proto, loc, glo, 'interaction', 'interaction')
        S.contrast('mh', proto, loc_alt, glo_alt, 'interaction (Pantomime-style global)', 'interaction_alt')
        for m in raw:
            S.test('mh', proto, m, raw[m], NAMES[m] + ' raw-DB', 'raw_vs_db')
    for m in ['pointnet_t_aug', 'pn2_t_aug', 'dgcnn_t_aug', 'p4t_aug', 'radhar', 'pointnet_t_tiny', 'p4t_tiny', 'pantomime_aug']:
        S.test('mh', 'LUO5', 'tvoxA', m, NAMES[m] + '-A (DB)', 'compute')
    # controls: gate->DBSCAN and fixed-scale normalisation
    V = {v: {m: variant(m, v) for m in raw} for v in ['gateDB', 'fixDB', 'fixraw']}
    for proto in PANT_PROTOS:
        g = V['gateDB']
        S.contrast('pant', proto, [(g[m], raw[m]) for m in LOCAL], [(g[m], raw[m]) for m in GLOBAL], 'interaction raw-vs-gateDB', 'ctl_gate')
        S.contrast('pant', proto, [(g[m], raw[m]) for m in LOCAL[:-1]], [(g[m], raw[m]) for m in GLOBAL + ['pantomime_aug']],
                   'interaction raw-vs-gateDB (PS global)', 'ctl_gate_alt')
        for m in g:
            S.test('pant', proto, m, g[m], NAMES[m] + ' gateDB-DB', 'ctl_gate')
            S.test('pant', proto, g[m], raw[m], NAMES[m] + ' raw-gateDB', 'ctl_gate')
        fd, fr = V['fixDB'], V['fixraw']
        S.contrast('pant', proto, [(fd[m], fr[m]) for m in LOCAL], [(fd[m], fr[m]) for m in GLOBAL], 'interaction FIXED raw-vs-DB', 'ctl_fix')
        S.contrast('pant', proto, [(fd[m], fr[m]) for m in LOCAL[:-1]], [(fd[m], fr[m]) for m in GLOBAL + ['pantomime_aug']],
                   'interaction FIXED (PS global)', 'ctl_fix_alt')
        S.contrast('pant', proto, [(fd[m], fr[m]) for m in LOCAL], [(fd[m], fr[m]) for m in GLOBAL if m not in ('tvoxA', 'radhar')],
                   'interaction FIXED point-only global', 'ctl_fix_pt')
        for m in fd:
            S.test('pant', proto, fd[m], fr[m], NAMES[m] + ' FIX raw-DB', 'ctl_fix')
    for a, b, l in [('abl_gatedb', 'p4t_tiny_gatedb', 'P4T-tiny-A (gateDB)'), ('abl_noden_fix', 'p4t_tiny_noden_fix', 'P4T-tiny-A (FIX raw)'),
                    ('abl_gatedb', 'pointnet_t_tiny_gatedb', 'PN+t-tiny-A (gateDB)'), ('abl_noden_fix', 'pointnet_t_tiny_noden_fix', 'PN+t-tiny-A (FIX raw)'),
                    ('tvoxA', 'abl_noden_fix', 'A FIXraw-perSampleDB'), ('p4t_aug', 'p4t_aug_noden_fix', 'P4T FIXraw-perSampleDB')]:
        S.test('pant', 'LUO5', a, b, l, 'ctl_compute')
    # gate-only for the remaining six models; fixed scale x {gate, gate->DBSCAN}
    gate = {m: variant(m, 'gate') for m in raw}
    for m in ['pn2_t_aug', 'dgcnn_t_aug', 'pantomime_aug', 'radhar', 'pointnet_t_tiny', 'p4t_tiny']:
        for proto in PANT_PROTOS:
            S.test('pant', proto, m, gate[m], NAMES[m] + ' gate-DB', 'gate')
            S.test('pant', proto, raw[m], gate[m], NAMES[m] + ' gate-raw', 'gate')
    gdb, fr = V['gateDB'], V['fixraw']
    fg, fgdb = ({m: variant(m, v) for m in raw} for v in ['fixgate', 'fixgateDB'])
    order = ['tvoxA', 'p4t_aug', 'pointnet_t_aug', 'pn2_t_aug', 'dgcnn_t_aug', 'pantomime_aug', 'radhar', 'pointnet_t_tiny', 'p4t_tiny']
    for proto in PANT_PROTOS:
        S.contrast('pant', proto, [(gdb[m], gate[m]) for m in LOCAL], [(gdb[m], gate[m]) for m in GLOBAL], 'interaction gate-vs-gateDB', 'ctl_gateonly')
        S.contrast('pant', proto, [(gdb[m], gate[m]) for m in LOCAL[:-1]], [(gdb[m], gate[m]) for m in GLOBAL + ['pantomime_aug']],
                   'interaction gate-vs-gateDB (PS global)', 'ctl_gateonly_alt')
        S.contrast('pant', proto, [(fgdb[m], fg[m]) for m in LOCAL], [(fgdb[m], fg[m]) for m in GLOBAL], 'interaction FIXED gate-vs-gateDB', 'ctl_fixgate')
        S.contrast('pant', proto, [(fgdb[m], fg[m]) for m in LOCAL[:-1]], [(fgdb[m], fg[m]) for m in GLOBAL + ['pantomime_aug']],
                   'interaction FIXED gate-vs-gateDB (PS global)', 'ctl_fixgate_alt')
        for m in order:
            S.test('pant', proto, fr[m], fg[m], NAMES[m] + ' FIX gate-raw', 'ctl_fixgate')
            S.test('pant', proto, fgdb[m], fg[m], NAMES[m] + ' FIX gate-gateDB', 'ctl_fixgate')
    for a, b, l in [('abl_gate_fix', 'p4t_tiny_gate_fix', 'P4T-tiny-A (FIX gate)'), ('abl_gate_fix', 'pointnet_t_tiny_gate_fix', 'PN+t-tiny-A (FIX gate)'),
                    ('p4t_aug_noden_fix', 'pn2_t_aug_gate_fix', 'PN++ FIX gate - P4T-lite FIX raw')]:
        S.test('pant', 'LUO5', a, b, l, 'ctl_compute')
    S.holm()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='stats.json')
    a = ap.parse_args()
    S = Stats(a.data, a.runs)
    arms = ['DB', 'raw', 'gate', 'gateDB', 'fixDB', 'fixraw', 'fixgate', 'fixgateDB']
    print('pooled accuracy (%), mean over seeds:', ' / '.join(arms))
    pooled = {}
    for m, n in NAMES.items():
        if m == 'tvoxF':
            continue
        for proto in PANT_PROTOS:
            v = [S.pooled('pant', variant(m, x), proto) for x in arms]
            pooled[f'{n}|{proto}'] = v
        print('%-16s' % n, '  '.join(proto + ' ' + '/'.join('--' if x is None else '%.1f' % x[0] for x in pooled[f'{n}|{proto}']) for proto in PANT_PROTOS))
    paper_family(S)
    print('family size', len(S.rows))
    for r in S.rows:
        print('%-16s %-4s %-13s %-40s n=%2d mean=%+6.2f CI[%+.2f,%+.2f] +%d/-%d pW=%.2g pPerm=%.2g Holm(W)=%.2g Holm(perm)=%.2g' % (
            r['fam'], r['ds'], r['proto'], r['label'], r['n'], r['mean'], *r['ci'], r['pos'], r['neg'], r['p_wil'], r['p_perm'], r['p_wil_holm'], r['p_perm_holm']))
    json.dump(dict(tests=S.rows, pooled=pooled), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
