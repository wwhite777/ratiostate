# Development-case scores, packed timing and coupling

This component recomputes 16 saved-prediction scores for the original four development images, six paired ratios from the previously recorded CPU timing rounds, retained cache payload counts, and the 24-row coupling decomposition. It does not rerun the model or benchmark timing. The full 252-cell replay is in `../fixed32/`.

Tested on Linux with Python 3.10.12 and NumPy 1.26.4. From the repository root, send output to a fresh absolute directory outside the repository:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -I -B saved/legacy/replay_all.py --output /replace/with/unused/absolute/legacy-output
```

The predictions, targets and coupling inputs are byte-identical to the archived bundle. Numerical routines are unchanged; the public adaptation updates reader guides, component manifests and one protocol-hash guard constant. `expected_scores.json`, `expected_timing.json` and `coupling/expected_rows.json` are independent saved comparisons, not the formula inputs. The score is the sampler-implied conditional metric, not a verified training loss. The six timing ratios describe the original CPU run; payload counts refer to retained backing-buffer bytes, not RSS or peak memory. Rights over upstream model weights or full MNIST data are not asserted.
