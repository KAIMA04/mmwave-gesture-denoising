"""Train and evaluate models on Pantomime or mHomeGes under the paper's protocols.

Every model uses the same recipe: AdamW (lr 2e-3, wd 1e-3), one-cycle schedule, batch 64, label smoothing 0.1.
Metrics go to <out>/<dataset>_results.jsonl and per-trial predictions to <out>/<dataset>_preds.jsonl;
runs whose key is already in the results file are skipped, so interrupted jobs can be resumed.

Example:
  python -m mmwave_gesture.train tvoxA,p4t_tiny_noden --dataset pantomime --data data --out runs --seeds 0,1,2
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .configs import M
from .models import make
from .pipeline import PackedData, voxelize, pointify
from .preprocess import DENOISERS

BS = 64
SUPP_SCENES = ['restaurant', 'industrial', 'occluded', 'multi_people']


def load(data_dir, cache_dir, dataset, part, den):
    """Load <dataset>_<part>.npz and apply (cached) per-trial outlier removal; adds cpts/coff."""
    d = np.load(os.path.join(data_dir, f'{dataset}_{part}.npz'), allow_pickle=False)
    d = {k: d[k] for k in d.files}
    if den == 'none':
        d['cpts'], d['coff'] = d['pts'].astype(np.float32), d['off']
        return d
    cf = os.path.join(cache_dir, f'clean_{dataset}_{part}_{den}.npz')
    if not os.path.exists(cf):
        f = DENOISERS[den]
        Ps = [f(d['pts'][a:b]) for a, b in zip(d['off'][:-1], d['off'][1:])]
        off = np.concatenate([[0], np.cumsum([len(p) for p in Ps])])
        np.savez(cf, pts=np.concatenate(Ps).astype(np.float32), off=off)
    c = np.load(cf)
    d['cpts'], d['coff'] = c['pts'], c['off']
    return d


class Runner:
    def __init__(self, cfg, nc, epochs, device):
        self.cfg, self.nc, self.epochs, self.dev = cfg, nc, epochs, device
        self.fix = None   # voxel box (lo, hi) for norm='fixed'
        self.fixp = None  # point (mean, std) for pnorm='fixed'

    def inputs(self, G, b, train, g, gs=None):
        cfg = self.cfg
        P, bid = G.batch(b)
        if cfg['kind'] in ('vox', 'radhar'):
            return voxelize(P, bid, len(b), cfg['n'], cfg['T'], cfg['norm'], train and cfg['aug'], g, cfg.get('cint', False), self.fix)
        return pointify(P, bid, len(b), train and cfg['aug'], g, gs, cfg.get('pnorm', 'sample'), self.fixp)

    def fit_fixed(self, G, idx):
        """Fixed-scale statistics from the training split only, on the input the model receives."""
        cfg = self.cfg
        if cfg.get('norm') != 'fixed' and cfg.get('pnorm') != 'fixed':
            return
        Pt, _ = G.batch(torch.tensor(np.asarray(idx), device=self.dev))
        x = Pt[:, :3]
        if cfg.get('norm') == 'fixed':
            q = torch.quantile(x.float().cpu(), torch.tensor([0.005, 0.995]), dim=0).to(self.dev)
            self.fix = (q[0][None], q[1][None])
        if cfg.get('pnorm') == 'fixed':
            self.fixp = (x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6))

    def train(self, G, y, idx, seed):
        self.fit_fixed(G, idx)
        torch.manual_seed(seed)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        rs = np.random.RandomState(seed)
        m = make(self.cfg, self.nc).to(self.dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
        steps = self.epochs * math.ceil(len(idx) / BS)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        Y = torch.tensor(y, device=self.dev)
        for _ in range(self.epochs):
            m.train()
            perm = rs.permutation(idx)
            for i in range(0, len(perm), BS):
                b = torch.tensor(perm[i:i + BS], device=self.dev)
                if len(b) < 2:
                    sch.step()
                    continue
                opt.zero_grad()
                crit(m(self.inputs(G, b, True, g)), Y[b]).backward()
                opt.step()
                sch.step()
        return m.eval()

    @torch.no_grad()
    def predict(self, m, G, idx):
        gs = torch.Generator(device=self.dev).manual_seed(1234)  # fixed test-time point sampling
        out = []
        for i in range(0, len(idx), 128):
            b = torch.tensor(idx[i:i + 128], device=self.dev)
            out.append(m(self.inputs(G, b, False, None, gs)).argmax(1).cpu().numpy())
        return np.concatenate(out)


def metrics(y, p):
    return dict(acc=accuracy_score(y, p), f1=f1_score(y, p, average='macro'), n=len(y))


def protocol_jobs(dataset, d, protocols):
    """(key suffix with {s} for the seed, train indices, test indices)."""
    y, g = d['gesture'], d['user']
    jobs = []
    if 'luo5' in protocols:
        folds = GroupKFold(5).split(np.zeros(len(y)), y, g)
        jobs += [(f'LUO5|{{s}}|{k}', a, b) for k, (a, b) in enumerate(folds)]
    if dataset == 'pantomime' and 'xenv' in protocols:
        env = d['env']
        for tr, te in [('office', 'open'), ('open', 'office')]:
            a = np.where(env == tr)[0]
            b = np.where((env == te) & ~np.isin(g, g[a]))[0]
            jobs.append((f'{tr}->{te}|{{s}}', a, b))
    if dataset == 'mhomeges' and 'xdist' in protocols:
        dv = np.array([float(x.rstrip('m')) for x in d['dist']])
        jobs.append(('near->far|{s}', np.where(dv <= 1.8)[0], np.where(dv >= 2.4)[0]))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', help='comma-separated configuration names from configs.py')
    ap.add_argument('--dataset', default='pantomime', choices=['pantomime', 'mhomeges'])
    ap.add_argument('--data', default='data', help='folder with the .npz files from prepare_*.py')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--protocols', default='luo5,xenv,supp', help='pantomime: luo5,xenv,supp; mhomeges: luo5,xdist')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-trials', type=int, default=0, help='truncate train/test index lists (quick tests only)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    protocols = a.protocols.split(',')
    res_f = os.path.join(a.out, f'{a.dataset}_results.jsonl')
    pred_f = os.path.join(a.out, f'{a.dataset}_preds.jsonl')
    done = {json.loads(l)['key'] for l in open(res_f)} if os.path.exists(res_f) else set()

    def log(r, ii, p):
        with open(pred_f, 'a') as f:
            f.write(json.dumps(dict(key=r['key'], idx=[int(v) for v in ii], pred=[int(v) for v in p])) + '\n')
        with open(res_f, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(r, flush=True)

    cut = (lambda x: x[:a.max_trials]) if a.max_trials else (lambda x: x)
    for name in a.models.split(','):
        cfg = M[name]
        d = load(a.data, a.out, a.dataset, 'primary', cfg['den'])
        y = d['gesture']
        nc = int(y.max()) + 1
        G = PackedData(d['cpts'], d['coff'], a.device)
        run = Runner(cfg, nc, a.epochs, a.device)
        jobs = protocol_jobs(a.dataset, d, protocols)
        for seed in seeds:
            for kt, tr, te in jobs:
                key = f'{name}|' + kt.format(s=seed)
                if key in done:
                    continue
                tr, te = cut(tr), cut(te)
                t = time.time()
                m = run.train(G, y, tr, seed)
                p = run.predict(m, G, te)
                log(dict(key=key, **metrics(y[te], p), sec=round(time.time() - t, 1)), te, p)
            if a.dataset == 'pantomime' and 'supp' in protocols and not any(k.startswith(f'{name}|primary->supp|{seed}|') for k in done):
                S = load(a.data, a.out, a.dataset, 'supp', cfg['den'])
                GS = PackedData(S['cpts'], S['coff'], a.device)
                m = run.train(G, y, cut(np.arange(len(y))), seed)
                ps = run.predict(m, GS, np.arange(len(S['gesture'])))
                scenes = [(e, S['env'] == e) for e in SUPP_SCENES] + [('open_1m_allangles_speeds', (S['env'] == 'open') & (S['dist'] == '1'))]
                for e, sel in scenes:
                    ii = np.where(sel)[0]
                    log(dict(key=f'{name}|primary->supp|{seed}|{e}', **metrics(S['gesture'][ii], ps[ii])), ii, ps[ii])


if __name__ == '__main__':
    main()
