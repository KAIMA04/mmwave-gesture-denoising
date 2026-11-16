# mmwave-gesture-denoising

Code and per-user splits for the IEEE RadarConf 2027 submission

**Denoising mmWave Gesture Point Clouds: Normalisation-Dependent Effects and a Compute-Matched Voxel vs. Point Comparison**
Kai Ma (University of Liverpool; now with Duke Kunshan University)

## Contents

| Path | Content |
|---|---|
| `splits/pantomime_splits.json` | Pantomime primary set: users tested in each of the 5 leave-users-out (LUO5) folds; Office→Open and Open→Office cross-room splits (train users, unseen test users, trial counts) |
| `splits/pantomime_luo5_folds.csv` | One row per Pantomime user: LUO5 test fold and room |
| `splits/mhomeges_splits.json` | mHomeGes: users tested in each LUO5 fold; cross-distance split (train ≤1.8 m, test ≥2.4 m; users overlap) |
| `splits/mhomeges_luo5_folds.csv` | One row per mHomeGes user: LUO5 test fold |

The training and evaluation code for the experiments in the paper will be added to this repository.

**Not released:** this repository contains no radar data. The self-recorded IWR6843AOP case-study recordings are not released.

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
