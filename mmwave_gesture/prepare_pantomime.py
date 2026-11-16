"""Pack the released Pantomime dataset (Zenodo 10.5281/zenodo.4459969) into two .npz files.

Expected layout after unzipping: <root>/{primary_exp,supp_exp}/<env>/<dist>/<angle>/<speed>/<user>/<gesture>/<trial>.pkl
Each .pkl is a list of frames; each frame is an (n, 4) array (x, y, z, intensity); frames without intensity get NaN.
Output per part: pts (N, 5) = x, y, z, intensity, frame index; off (trials + 1) offsets into pts;
and per-trial metadata exp, env, dist, angle, speed, user, gesture (0-based), trial.

Usage: python -m mmwave_gesture.prepare_pantomime --root PATH/dataset_5_11_2020 --out data
"""
import argparse, glob, os, pickle
import numpy as np

META = ['exp', 'env', 'dist', 'angle', 'speed', 'user', 'gesture', 'trial']


def _xyzi(fr):
    fr = np.asarray(fr, dtype=np.float64)
    if fr.shape[1] == 3:  # a few frames carry no intensity column
        fr = np.c_[fr, np.full(len(fr), np.nan)]
    return fr[:, :4]


def pack(root, exp, out):
    files = sorted(glob.glob(os.path.join(root, exp, '*', '*', '*', '*', '*', '*', '*.pkl')))
    pts, off, meta = [], [0], {k: [] for k in META}
    for f in files:
        frames = pickle.load(open(f, 'rb'), encoding='latin1')
        P = [np.c_[_xyzi(fr), np.full(len(fr), t)] for t, fr in enumerate(frames) if len(fr)]
        P = np.concatenate(P) if P else np.zeros((0, 5))
        pts.append(P); off.append(off[-1] + len(P))
        parts = os.path.relpath(f, root).split(os.sep)
        for k, v in zip(META, parts[:7] + [parts[7][:-4]]):
            meta[k].append(v)
    meta['gesture'] = [int(g) - 1 for g in meta['gesture']]
    name = 'primary' if exp == 'primary_exp' else 'supp'
    np.savez_compressed(os.path.join(out, f'pantomime_{name}.npz'), pts=np.concatenate(pts).astype(np.float32),
                        off=np.array(off), **{k: np.array(v) for k, v in meta.items()})
    print(f'pantomime_{name}: {len(files)} trials, {off[-1]} points')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='unzipped Pantomime dataset folder')
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for exp in ['primary_exp', 'supp_exp']:
        pack(a.root, exp, a.out)
