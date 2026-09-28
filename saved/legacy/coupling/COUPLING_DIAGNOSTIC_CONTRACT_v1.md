# Fixed-feature readout decomposition protocol

This is a retrospective descriptive analysis of four existing development-image traces, specified before its decomposition values were calculated. It is not a new full-model run or a confirmation sample. The independent sequence count is four. Retain all 783 inclusive positions, batch 0, layers 0 and 7, head 0, 32 output coordinates, and all three fixed-feature arms (`fp32`, `bf16`, `bf16_scale_half`), yielding 24 rows. Full-model boundary-arm trajectories are different and are not decomposed with native q/k/v.

The reference promotes the saved FP32 mapped q/k/v to FP64, performs inclusive sequential FP64 outer-product/state accumulation, and then FP64 query contractions. At each position, let `N=qᵀS64`, `D=qᵀz64+epsilon`, and `y=N/D`. Epsilon is the exact layer-specific additive constant, not a floor. Require finite inputs and strictly positive finite `D`. Compare `N/D` with the saved FP64 reference at `2e-13` absolute plus `2e-13` relative tolerance. This FP64 reference is not exact real arithmetic.

For each arm, promote its saved FP32 numerator, effective denominator, and output to FP64. Undo representation scale before perturbation: `scale=1` for fp32/bf16 and `scale=0.5` for `bf16_scale_half`; `Nhat=N_saved/scale`, `Dhat=D_saved/scale`. The saved effective denominator already contains the scaled additive epsilon. The unchanged saved output is `yhat`. Require finite `Nhat`, `Dhat`, `yhat` and `Dhat>0`. Define coordinatewise

```
a = (Nhat - N)/Dhat
b = -y*(Dhat - D)/Dhat
c = yhat - Nhat/Dhat
e = yhat - y
```

Then `e=a+b+c` up to evaluated FP64 rounding. `a` and `b` include storage and contraction error; they are not isolated storage interventions. `c` is the final saved FP32 reciprocal/multiply residual relative to the idealized ratio of recorded contractions. These local terms do not identify full-sequence task-score causes.

For every image/layer/head/arm row, report raw and reference-normalized L2 norms of `a`, `b`, `a+b`, `c`, and `e`; maximum absolute identity residual; reference-match error; denominator minima; and counts. If `||y||₂=0`, normalized quantities are null with a reason. The literal componentwise cancellation fraction is `C1=1-sum(abs(a+b))/(sum(abs(a))+sum(abs(b)))`. When its denominator is exactly zero, C1 is null with a zero-contribution flag. Record all three absolute sums. Do not clip computed C1; allow at most `1e-12` endpoint rounding. L2 triangle-bound slack is not a cancellation fraction.

Retain all eight BF16 selected traces, plus fp32 and half-scale rows. Verify that normalized half-scale `a/b/c/e/C1` match unscaled BF16 counterparts. No outcome-dependent filtering, pooled population estimate, p-value, or confidence interval is defined. Required fixtures include positive and negative proportional changes, numerator-only and reinforcing terms, orthogonal terms, zero-contribution undefined C1, finite output-rounding residual, invalid/zero denominator rejection, and half-scale normalization. Identity tolerance is `max|e-(a+b+c)| <= 1e-12*max(1,max|e|,max|y|,max|yhat|)`; relative output error is compared with prior control metrics at `2e-13` absolute/relative tolerance. These are reproduction tolerances, not practical-effect margins.

The accompanying `replay.py` implements these rules on the bundled authored derived arrays. The original frozen development contract SHA-256 is `a1b1afd69212fdd5be02aa6d75ca7d1c35d561b237d3a41c953372b484300deb`; this public-facing protocol removes only internal workflow directions while retaining the numerical definitions. The distributed protocol identity is checked by the adjacent source and component manifest; the original full-contract hash above is lineage only. Neither original nor distributed local hash authenticates an external preregistration timestamp.
