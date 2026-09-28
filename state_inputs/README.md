# Selected normalizer states

This directory distributes the normalizer arrays needed for the retrospective diagnostics from 192 fixed32 image–arm runs. Each set covers 783 returned and next-call feed normalizers for layers 0 and 7, head 0, 32 coordinates. Stored-component arrays are included where applicable; their numerical dtype is described by the original metadata.

`runs/*/trace.json`, `manifest.json` and `complete.json` are preserved historical records of the original larger runs. They also name matrix-state, prediction and other arrays **not distributed in this subset**. They must not be treated as inventories of the files included here. `FILES.sha256` is the exact inventory of the distributed subset; from this directory, `sha256sum --check FILES.sha256` verifies it. Saved prediction parameters and pixels are distributed separately under `../saved/fixed32/`.

The retained normalizer arrays support the complete retrospective command in `../analysis/README.md`. Fixed-feature matrix/normalizer decomposition inputs are under `../saved/legacy/coupling/`; raw matrix-state trajectories for the 192 runs are not included. Neither this subset nor its historical metadata is a complete model checkpoint. Model graph and weights are acquired separately.
