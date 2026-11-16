"""CPU smoke test on tiny synthetic data: every model family builds, trains one epoch and predicts.

Run from the repository root: python tests/smoke_test.py
"""
import os, subprocess, sys, tempfile
import numpy as np


def synth(path, n_trials, users, envs, rng, supp=False):
    pts, off = [], [0]
    meta = {k: [] for k in ['env', 'dist', 'angle', 'speed', 'user', 'gesture', 'trial', 'exp']}
    for i in range(n_trials):
        n = rng.integers(20, 60)
        P = np.c_[rng.normal([1.2, 0, 0], 0.1, (n, 3)), rng.random(n), np.sort(rng.integers(0, 30, n))]
        P[:5, :3] = rng.normal(0, 0.02, (5, 3))  # near-field cluster at the radar
        pts.append(P)
        off.append(off[-1] + n)
        meta['env'].append(envs[i % len(envs)]); meta['dist'].append('1'); meta['angle'].append('0'); meta['speed'].append('normal')
        meta['user'].append(str(users[i % len(users)])); meta['gesture'].append(i % 3); meta['trial'].append(str(i))
        meta['exp'].append('supp_exp' if supp else 'primary_exp')
    np.savez(path, pts=np.concatenate(pts).astype(np.float32), off=np.array(off), **{k: np.array(v) for k, v in meta.items()})


def main():
    rng = np.random.default_rng(0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        data, out = os.path.join(tmp, 'data'), os.path.join(tmp, 'runs')
        os.makedirs(data)
        # users 1-5 in the office, 6-10 in the open room
        synth(os.path.join(data, 'pantomime_primary.npz'), 100, list(range(1, 11)), ['office'] * 5 + ['open'] * 5, rng)
        synth(os.path.join(data, 'pantomime_supp.npz'), 30, [101, 102], ['restaurant', 'industrial', 'occluded', 'multi_people', 'open'], rng, supp=True)
        H = dict(np.load(os.path.join(data, 'pantomime_primary.npz')))
        H['dist'] = np.array(['1.2m', '2.4m'] * 50)
        np.savez(os.path.join(data, 'mhomeges_primary.npz'), **H)
        models = 'tvoxA,abl_gatedb_fix,radhar_noden,pointnet_t_tiny_gate,dgcnn_t_aug_fix,pn2_t_aug_noden_fix,p4t_tiny_gatedb,pantomime_aug,abl_int'
        common = [sys.executable, '-m', 'mmwave_gesture.train', models, '--data', data, '--out', out, '--epochs', '1', '--device', 'cpu']
        subprocess.run(common + ['--dataset', 'pantomime', '--protocols', 'luo5,xenv,supp'], cwd=root, check=True)
        subprocess.run(common + ['--dataset', 'mhomeges', '--protocols', 'luo5,xdist'], cwd=root, check=True)
        lines = open(os.path.join(out, 'pantomime_results.jsonl')).read().splitlines()
        assert len(lines) == 9 * (5 + 2 + 5), len(lines)
        assert len(open(os.path.join(out, 'mhomeges_results.jsonl')).read().splitlines()) == 9 * 6
        sys.path.insert(0, root)
        from mmwave_gesture.stats import Stats
        S = Stats(data, out)
        assert S.failures.shape == (100,)
        print('smoke test passed:', len(lines), 'Pantomime runs')


if __name__ == '__main__':
    main()
