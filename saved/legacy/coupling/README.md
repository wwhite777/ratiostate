# Fixed-feature coupling replay

The adjacent `replay.py` recomputes the 24 retrospective readout-decomposition rows and 13 term arrays from four development traces. Its scientific definitions and tolerances are in `COUPLING_DIAGNOSTIC_CONTRACT_v1.md`, a curated public protocol preserving the numerical definitions. The adjacent analyzer checks the distributed protocol hash; only that hash constant was adapted for the public text. This analysis is descriptive; it does not isolate storage-only causal effects or establish a new prospective test.

From the repository root, run `saved/legacy/replay_all.py` as directed in `saved/legacy/README.md`; it invokes this component with the same two-thread CPU environment. No model graph or training data are required. The included derived trace arrays are released here with attribution and without a claim to license upstream model weights or MNIST data. Hashes attest file identity, not an independently authenticated preregistration timestamp.
