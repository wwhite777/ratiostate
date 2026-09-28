# Supplementary scores, contrasts and selected-state summaries

These files contain the reported machine-readable score, contrast, position and selected-state summaries for the RatioState case study. They contain derived numeric results and image identifiers; the separate `saved/` and `state_inputs/` directories provide the executable replay inputs. The author authorized publication of this curated export. The MIT code license does not grant rights to upstream model weights or the full MNIST dataset.

`absolute_scores.csv` has all 252 complete score cells: 60 from ten purposively selected development images and 192 from a separate, fixed 32-image within-model evaluation. Each image contributes six arms. `review_scope=legacy_context` identifies the original four images under A00, A11, FP16 and the two-word BF16 pair (16 cells); the later independent saved-prediction reconstruction covered all 252 cells, including 236 beyond the original 16. That review did not rerun graph trajectories. Cohorts must not be pooled or treated as independent models.

The score column `conditional_bits_per_predicted_pixel` is the idealized sampler-implied conditional negative log score for 783 predicted pixels after conditioning on pixel 0; lower is better. It is not verified to be the original training loss. `index` is the original MNIST test index and `label` its digit label. The development designation refers to its use in this study, regardless of the original dataset split. `A00` stores matrix and normalizer states in FP32; `A10` stores the matrix in BF16 and normalizer in FP32; `A01` stores the matrix in FP32 and normalizer in BF16; `A11` stores both in BF16. `fp16_boundary` stores both in IEEE binary16. `bf16_pair_boundary` stores both as a two-word BF16 residual representation with an FP32-sized retained payload. All arms use FP32 graph arithmetic within a call.

`fixed32_contrasts.csv` has nine paired differences for each of the 32 evaluation images. Positive means the first named condition has the larger, worse score. The definitions are:

| Contrast | Score difference |
| --- | --- |
| normalizer_given_bf16_matrix | A11 − A10 |
| normalizer_only | A01 − A00 |
| interaction | A11 − A10 − A01 + A00, evaluated left to right |
| A10_minus_fp16 | A10 − fp16_boundary |
| matrix_only | A10 − A00 |
| both | A11 − A00 |
| matrix_given_bf16_normalizer | A11 − A01 |
| fp16_minus_native | fp16_boundary − A00 |
| pair_minus_native | bf16_pair_boundary − A00 |

`fixed32_contrast_summary.csv` gives the saved `n_images`, mean, sample standard deviation, median, minimum, maximum, exact positive/negative/zero counts and pointwise 95% image-resampling bounds for each contrast. `fixed32_arm_summary.csv` gives the analogous saved absolute-score summaries for the six arms. The bounds came from 20,000 image-level resamples with PCG64 seed 20260924 reset per series and linear 0.025/0.975 quantiles; this export did not rerun resampling. These original exports preserve the historical descriptive reporting. V3 adds a separately labeled conditional finite-pool interpretation and exploratory paired tests in successor files; original values and protocol wording are unchanged. No simultaneous-coverage or equivalence claim is made.

`development_state_role_contrasts.csv` gives six state-role differences on each of the ten development images. `matrix` = A10 − A00, `normalizer` = A01 − A00, `both` = A11 − A00, `interaction` = A11 − A10 − A01 + A00, `matrix_given_bf16_normalizer` = A11 − A01, and `normalizer_given_bf16_matrix` = A11 − A10. `development_state_role_summary.csv` contains their saved means, ranges and exact sign counts; no confidence bounds were assigned to this purposive cohort. In every contrast file, `difference_bits_per_predicted_pixel` is the paired score difference for one image. Values are written as round-trip decimal representations of the saved finite binary64 numbers, without publication-table rounding.

The study uses one trained image model. Training-data overlap, near duplication, practical quality tolerance, wider-model generality, and correspondence between this score and the original training objective remain unresolved. Storage precision changes can alter later features and trajectories; these score tables alone do not isolate a causal readout mechanism or measure peak memory and throughput. All cells are present here; an absent or invalid cell would require an explicit missingness record, never a zero score. Scientific source identities and the curated protocol extracts are summarized in `../protocols/`; saved executable components retain file-level manifests.

## Retrospective additions in revision2

The exploratory subdirectory contains absolute score deviations, positional score summaries and selected-normalizer event summaries computed on 26 September after review feedback. The original CSVs above remain byte-identical. All 252 scores were reconstructed in the later computational review; 236 were beyond the original 16. The new analyses do not amend the original protocol. The reproduction companion distinguishes saved-prediction replay, selected-state replay and full graph execution.


## V3 additions

The v3 statistics use the exact original 192 fixed32 score cells to report signed shifts, SD, SE and explicitly exploratory paired tests. Cumulative storage-error outputs separate coordinate and image-level mass ratios, upward/downward rounding and absorbed-positive mass. Position windows also report per-prediction rates with their actual lengths. See the repository root README and `analysis/README.md` for retained raw/feed arrays, predictions and executable replay entry points. The large `exploratory_v3/state_coordinate_ratios_v2.npz` is omitted from Git and regenerated by the included analyzer.
