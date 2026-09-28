# Retrospective saved-state and position analysis

`analyze_saved_v2.py` is the unchanged saved-data analyzer. It uses the published 192 fixed32 score cells, 32×6×783 token-score array, selected raw/feed normalizer states, and saved FP16 range diagnostics. It regenerates position, mass, endpoint and coordinate-ratio outputs, including the omitted `state_coordinate_ratios_v2.npz` (~93.6 MB). This is retrospective descriptive work, not a new prospective sample or a model run.

With Linux/Python 3.10.12, NumPy 1.26.4, mpmath 1.3.0, one numerical thread, and an unused absolute output path outside the repository, run from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -I -B analysis/analyze_saved_v2.py --analysis-json analysis/inputs/analysis_v3.json --absolute-scores-csv results/absolute_scores.csv --token-npz analysis/inputs/token_bits_v2.npz --runs-dir state_inputs/runs --range-json analysis/inputs/natural_check_v1.json --output-dir /replace/with/unused/absolute/state-analysis-output
```

`analysis/inputs/analysis_v3.json` and `natural_check_v1.json` preserve original numerical identities; local hashes do not authenticate external preregistration timestamps. The repository includes every selected state file needed by this command. Scientific boundaries and missingness are described in `../protocols/METHODS.md` and `../results/README.md`.
