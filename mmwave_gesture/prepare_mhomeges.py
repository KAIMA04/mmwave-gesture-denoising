"""Segment the mHomeGes recordings (github.com/GestureMan/mHomeGes-dataset) into gesture repetitions.

Each CSV (longGes_<dist>/<user>/point_<user>_<dist>_<gesture>.csv) is a continuous recording.
A new repetition starts where consecutive non-empty frame indices differ by >= 8;
segments with < 3 non-empty frames or < 15 points are dropped (24,105 samples remain).
Output: data/mhomeges_primary.npz with pts (N, 5) = x, y, z, Doppler, frame; off; gesture, user, env, dist, src.

Usage: python -m mmwave_gesture.prepare_mhomeges --root PATH/mHomeGes-dataset-master --out data
"""
import argparse, collections, glob, os, re
import numpy as np
import pandas as pd

CLASSES = ['circle', 'clap', 'down', 'knock', 'lift', 'pull', 'push', 'up', 'yawn', 'z']
GAP, MIN_FRAMES, MIN_POINTS = 8, 3, 15


def main(root, out):
    S, drop = [], 0
    for f in sorted(glob.glob(os.path.join(root, 'longGes_*', '*', '*.csv'))):
        rel = os.path.relpath(f, root).replace(os.sep, '/')
        dist = rel.split('/')[0][len('longGes_'):]
        user = rel.split('/')[1]
        g = re.sub(r'[^a-z]', '', f.split('_')[-1][:-4]).replace('circcle', 'circle')
        if g not in CLASSES:
            continue
        d = pd.read_csv(f)
        fr = d['Frame #'].values
        X = d[['X', 'Y', 'Z', 'Doppler']].values
        u = np.unique(fr)
        seg = np.searchsorted(u[1:][np.diff(u) >= GAP], fr, side='right')
        for k in np.unique(seg):
            m = seg == k
            if len(np.unique(fr[m])) < MIN_FRAMES or m.sum() < MIN_POINTS:
                drop += 1
                continue
            S.append(dict(P=np.c_[X[m], fr[m]], gesture=CLASSES.index(g), user=user, env='home', dist=dist, src=rel))
    pts = np.concatenate([s['P'] for s in S]).astype(np.float32)
    off = np.concatenate([[0], np.cumsum([len(s['P']) for s in S])])
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'mhomeges_primary.npz'), pts=pts, off=off,
                        **{k: np.array([s[k] for s in S]) for k in ['gesture', 'user', 'env', 'dist', 'src']})
    print(f'mhomeges: {len(S)} samples ({drop} short segments dropped)', dict(collections.Counter(s["gesture"] for s in S)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='mHomeGes-dataset-master folder')
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    main(a.root, a.out)
