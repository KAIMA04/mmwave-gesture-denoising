# mmwave-gesture-denoising

Code and per-user splits for the IEEE RadarConf 2027 submission

**Denoising mmWave Gesture Point Clouds for Deep Networks: Normalisation-Dependent Effects and a Compute-Matched Voxel vs. Point Comparison**
Kai Ma (University of Liverpool; now with Duke Kunshan University)

## Contents

| Path | Content |
|---|---|
| `mmwave_gesture/prepare_pantomime.py` | Packs the released Pantomime `.pkl` files into `data/pantomime_{primary,supp}.npz` |
| `mmwave_gesture/prepare_mhomeges.py` | Segments the mHomeGes recordings into repetitions, written to `data/mhomeges_primary.npz` |
| `mmwave_gesture/preprocess.py` | Outlier removal: DBSCAN largest cluster, near-field gate, gate→DBSCAN, radius outlier removal, grid connected components |
| `mmwave_gesture/pipeline.py` | Augmentation, temporal voxelisation (per-sample, isotropic or fixed-scale), point resampling (per-sample or fixed-scale z-score) |
| `mmwave_gesture/models.py` | TVoxNet and the re-implemented baselines (see below) |
| `mmwave_gesture/configs.py` | Every model / input configuration used in the paper |
| `mmwave_gesture/train.py` | Training and evaluation under LUO5, cross-room, primary→supplementary (Pantomime), and LUO5, cross-distance (mHomeGes) |
| `mmwave_gesture/stats.py` | User-level Wilcoxon and sign-flip permutation tests, bootstrap CIs, Holm correction, pooled accuracies |
| `mmwave_gesture/benchmark.py` | Parameters, MACs and single-thread CPU latency (PyTorch and ONNX Runtime) |
| `splits/` | Per-user splits as JSON and CSV (see below) |
| `tests/smoke_test.py` | CPU smoke test on synthetic data |

**Not released:** this repository contains no radar data and no results. The self-recorded IWR6843AOP case-study recordings are not released.

The baselines are our own lighter PyTorch re-implementations, not the original code:

- PointNet+t and PointNet+t-tiny: PointNet, Qi et al., CVPR 2017.
- PointNet++ SSG: Qi et al., NeurIPS 2017.
- DGCNN-lite+t: Wang et al., ACM TOG 2019.
- P4Transformer-lite and P4T-tiny: after Fan et al., CVPR 2021.
- Pantomime-style: after Palipana et al., IMWUT 2021.
- RadHAR: after Singh et al., mmNets 2019.

Please cite the original papers when you use them.

## Usage

```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # quick CPU check

# 1. Data: download the datasets (see below), then
python -m mmwave_gesture.prepare_pantomime --root PATH/dataset_5_11_2020 --out data
python -m mmwave_gesture.prepare_mhomeges --root PATH/mHomeGes-dataset-master --out data

# 2. Training and evaluation: 3 seeds, all protocols; results and per-trial predictions are written to runs/
python -m mmwave_gesture.train tvoxA,abl_noden,p4t_tiny,p4t_tiny_noden --dataset pantomime --seeds 0,1,2
python -m mmwave_gesture.train tvoxA,abl_noden --dataset mhomeges --protocols luo5,xdist --seeds 0,1,2

# 3. User-level statistics: needs all configurations of the paper (see stats.py) for 3 seeds; 302 tests, Holm over all
python -m mmwave_gesture.stats --data data --runs runs --out stats.json

# 4. Complexity and latency
python -m mmwave_gesture.benchmark --data data --out latency.json
```

Configuration names (`configs.py`):

- `tvoxA`: TVoxNet-A; `tvoxF`: the wide variant; `abl_*`: TVoxNet ablations.
- Baselines: `pointnet_t_aug`, `pn2_t_aug`, `dgcnn_t_aug`, `p4t_aug`, `pantomime_aug`, `radhar`, `pointnet_t_tiny`, `p4t_tiny`.
- Input suffixes on a baseline name:
  - `_noden`: raw points;
  - `_gate`: near-field gate only;
  - `_gatedb`: gate, then DBSCAN;
  - `_fix`, `_noden_fix`, `_gate_fix`, `_gatedb_fix`: the same inputs with fixed-scale normalisation.
- TVoxNet-A input variants: `abl_noden`, `abl_gate`, `abl_gatedb`, `tvoxA_fix`, `abl_noden_fix`, `abl_gate_fix`, `abl_gatedb_fix`.

Training uses AdamW, a one-cycle schedule, 60 epochs, batch 64 and label smoothing 0.1 for every model. A GPU is recommended; the paper's runs used NVIDIA A40 GPUs.

## Splits

| File | Content |
|---|---|
| `splits/pantomime_splits.json` | Pantomime primary set: users tested in each of the 5 leave-users-out (LUO5) folds; Office→Open and Open→Office splits (train users, unseen test users, trial counts) |
| `splits/pantomime_luo5_folds.csv` | One row per Pantomime user: LUO5 test fold and room |
| `splits/mhomeges_splits.json` | mHomeGes: users tested in each LUO5 fold; cross-distance split (train ≤1.8 m, test ≥2.4 m; users overlap) |
| `splits/mhomeges_luo5_folds.csv` | One row per mHomeGes user: LUO5 test fold |

`train.py` derives the same folds itself, using `GroupKFold(5)` over the trial order written by the `prepare_*.py` scripts.

## Obtaining the datasets

Both datasets must be downloaded from their original sources, under their own licences:

- **Pantomime** (Palipana et al., IMWUT 5(1), 2021): Zenodo, doi:[10.5281/zenodo.4459969](https://doi.org/10.5281/zenodo.4459969), CC BY 4.0.
  - We use the primary set (9,011 trials, 42 user IDs, office and open rooms).
  - User IDs in our split files are those of the released data.
- **mHomeGes** (Liu et al., IMWUT 4(4), 2020, doi:[10.1145/3432235](https://doi.org/10.1145/3432235)): https://github.com/GestureMan/mHomeGes-dataset.
  - User IDs (800–825) are the user folder names of the released data.

## Protocol notes

- **LUO5.** Folds are scikit-learn `GroupKFold(5)` grouped by user. We also list the resulting user sets explicitly, so the splits do not depend on the trial order.
- **Pantomime cross-room.** Train on all trials of one room. Test on the other room's trials from users who do not appear in the training room.
- **mHomeGes segmentation.** The recordings are continuous, so we split them into repetitions:
  - a new repetition starts where consecutive non-empty frame indices differ by ≥8;
  - segments with fewer than 3 non-empty frames or fewer than 15 points are dropped.

  This leaves 24,105 samples.
- **mHomeGes cross-distance.** Train on 1.2–1.8 m, test on 2.4–3.0 m. The distances 1.95–2.25 m are not used.

## Licence

MIT (see `LICENSE`). This licence covers the contents of this repository only, not the datasets.
