"""Parameters, MACs and single-thread CPU latency (batch 1) of the networks and the preprocessing.

Network: PyTorch eager and ONNX Runtime (one intra/inter-op thread), 20 warm-up + 300 timed runs,
median over 5 repeats. Preprocessing: per-trial denoising + numpy voxelisation and point resampling on
300 random Pantomime primary trials.

Usage: python -m mmwave_gesture.benchmark --data data --out latency.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, json, platform, tempfile, time
import numpy as np
import torch

from .configs import M
from .models import make
from .pipeline import pointify
from .preprocess import dbscan, gridcc, ror, gate

MODELS = ['tvoxA', 'tvoxF', 'radhar', 'pointnet_t_aug', 'dgcnn_t_aug', 'pn2_t_aug', 'p4t_aug', 'pantomime_aug', 'pointnet_t_tiny', 'p4t_tiny']


def timeit(f, x, n=300, w=20, rep=5):
    v = []
    for _ in range(rep):
        for _ in range(w):
            f(x)
        t = time.perf_counter()
        for _ in range(n):
            f(x)
        v.append((time.perf_counter() - t) / n * 1000)
    return float(np.median(v))


def vox_np(P, n=16, T=8):
    xyz, fr = P[:, :3], P[:, 4]
    mn = xyz.min(0)
    ext = np.maximum(xyz.max(0) - mn, 1e-9)
    ii = np.clip(np.floor((xyz - mn) / ext * n).astype(int), 0, n - 1)
    lo, hi = fr.min(), fr.max()
    ti = np.clip(np.floor((fr - lo) / max(hi - lo, 1e-9) * T).astype(int), 0, T - 1)
    v = np.zeros((T, n, n, n), np.float32)
    np.add.at(v, (ti, ii[:, 0], ii[:, 1], ii[:, 2]), 1)
    return np.log1p(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='latency.json')
    ap.add_argument('--runs', type=int, default=300)
    ap.add_argument('--models', default=','.join(MODELS))
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    res = {'_env': dict(cpu=platform.processor(), torch=torch.__version__, python=platform.python_version(), threads=1, batch=1)}
    try:
        from thop import profile
    except ImportError:
        profile = None
    try:
        import onnxruntime as ort
        res['_env']['onnxruntime'] = ort.__version__
    except ImportError:
        ort = None
    tmp = tempfile.mkdtemp()
    for name in a.models.split(','):
        cfg = M[name]
        torch.manual_seed(0)
        m = make(cfg, 21).eval()
        x = torch.randn(1, 256, 4) if cfg['kind'] == 'pts' else torch.rand(1, cfg['T'], cfg['n'], cfg['n'], cfg['n'])
        r = dict(params=sum(p.numel() for p in m.parameters()))
        if profile is not None:
            r['MMACs'] = round(profile(make(cfg, 21).eval(), inputs=(x,), verbose=False)[0] / 1e6, 2)
        with torch.no_grad():
            r['torch_ms'] = round(timeit(m, x, a.runs), 3)
        if ort is not None:
            p = os.path.join(tmp, f'{name}.onnx')
            torch.onnx.export(m, (x,), p, opset_version=17, dynamo=False, input_names=['x'], output_names=['y'])
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            s = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
            r['onnx_ms'] = round(timeit(lambda v: s.run(None, {'x': v}), x.numpy(), a.runs), 3)
        res[name] = r
        print(name, r, flush=True)
    d = np.load(os.path.join(a.data, 'pantomime_primary.npz'))
    off, pts = d['off'], d['pts']
    ids = np.random.RandomState(0).choice(len(off) - 1, 300, replace=False)
    pre = {}
    for nm, f in [('dbscan', dbscan), ('ror', ror), ('grid', gridcc), ('gate', gate), ('none', lambda P: P)]:
        ts, tv = [], []
        for i in ids:
            P = pts[off[i]:off[i + 1]]
            t = time.perf_counter()
            Q = f(P)
            t1 = time.perf_counter()
            vox_np(Q)
            ts.append(t1 - t)
            tv.append(time.perf_counter() - t1)
        pre[nm] = dict(denoise_ms_mean=round(np.mean(ts) * 1000, 3), denoise_ms_p95=round(np.percentile(ts, 95) * 1000, 3),
                       voxelize_ms_mean=round(np.mean(tv) * 1000, 3))
    for nm, f in [('pointify_dbscan', dbscan), ('pointify_raw', lambda P: P)]:
        tp = []
        for i in ids:
            Q = torch.tensor(f(pts[off[i]:off[i + 1]]), dtype=torch.float32)
            Q[:, 3] = torch.nan_to_num(Q[:, 3])
            bid = torch.zeros(len(Q), dtype=torch.long)
            gs = torch.Generator().manual_seed(0)
            t = time.perf_counter()
            pointify(Q, bid, 1, False, None, gs)
            tp.append(time.perf_counter() - t)
        pre[nm + '_ms_mean'] = round(np.mean(tp) * 1000, 3)
    res['_preprocess'] = pre
    print(pre)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
