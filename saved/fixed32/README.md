# Saved-prediction score replay

This component recomputes all 252 conditional score cells from the included prediction parameters and selected 784-pixel sequences: 60 development cells and 192 fixed32 cells. It also computes nine contrasts per image and the original pointwise fixed32 bootstrap (20,000 PCG64 resamples, seed 20260924). This is saved-data replay, not graph inference or a training-loss verification.

From the repository root, use the tested Linux/Python 3.10.12, NumPy 1.26.4 environment. Give `--output` a fresh absolute path outside the repository:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -I -B saved/fixed32/replay.py --output /replace/with/unused/absolute/saved-score-output
```

The original numerical source, scorer, predictions and pixels retain their archived bytes. `package_manifest.json` was updated only for this reader-facing README. Cohorts remain separate; one image is the unit. A missing cell requires an explicit reason and is never replaced with zero. The score is the sampler-implied discretized-mixture negative log2 probability per predicted pixel after conditioning on pixel 0; correspondence to training loss is unverified.
