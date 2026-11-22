# Branches B-JiT / C-consensus: results (diagnostics only, NOT official scores)

Scripts: `unlearning/branches/{bjc.py, BJ_jit.py, BC_consensus.py, BX_interp.py, bjc_sum.py}`.
Metrics are computed on the 3,500 forget images, using views clean/mild/strong/diag (seed 0) and `cm`. `cm` is a CutMix of random forget pairs, used as a pseudo-control proxy.
- `agree` = argmax agreement with the original on the same view.
- `kl` = KL(orig‖sub) at T=2.
- `cecl` = CE against the CLEAN pseudo-label (original argmax on the clean image).
- TEMPCTRL = calibration control. It is head weight+bias scaling (= global logit temperature), with the scale s matched to the candidate's clean CE. Its metrics are computed analytically from the cached original logits. Argmax is unchanged by construction.

Original (A0): clean CE 0.0003, conf 0.9997, 45 classes. mild conf 0.986, strong conf 0.910, diag conf 0.991.

## B-JiT
The faithful loss, checked against `src/lipschitz.py::modify_weight` in github jwf40/Information-Theoretic-Unlearning:
- SGD with no momentum, over all params.
- One pass over the forget set, in order, batch 128, one step per batch (28 steps).
- loss = (1/N) Σ_j Σ_{i∈batch} ‖f(x_i) − f(x_i+ξ_ij)‖₂ / ‖ξ_ij‖₂, where f = logits. Note that this is a SUM over the batch, not a mean.
- ξ ~ N(0, σ²) is added in normalised input space, then clamped to the batch [min, max].
- N = 25, and f(x+ξ) is computed under no_grad.
- Paper: lr 0.01, σ 0.5. Code: lr 0.1.

Implementation: clean branch in fp32/TF32, noisy branch in fp16 autocast. Each run takes about 33 s.

| run | drift (rel L2) | clean CE mean/sd | clean conf | clean agree | ncls | cm agree / kl | mild agree / kl | TEMPCTRL cm kl / mild kl |
|---|---|---|---|---|---|---|---|---|
| all lr0.1 σ.5 (code) | NaN | diverged (NaN at step 9) | | | | | | |
| all lr0.1 σ.25 / σ1.0 | NaN / collapse | | | | | | | |
| all lr0.01 σ.5 (paper) | 0.0092 | 28.0/8.4 | 0.979 | 0.00 | 1 | 0.003 / 11.7 | 0.00 / 13.8 | (s→0.001) |
| last4 lr0.1 | NaN | | | | | | | |
| gamma lr0.1 | 0.0196 | 11.4/4.1 | 0.47 | 0.007 | 4 | 0.005 / 5.09 | | |
| all lr1e-4 | 0.00073 | 4.40/3.43 | 0.666 | 0.21 | 78 | 0.128 / 2.55 | 0.167 / 3.15 | 2.93 / 4.25 |
| all lr3e-4 | 0.0014 | 6.99/2.72 | 0.491 | 0.016 | 35 | 0.026 / 3.41 | 0.014 / 4.70 | floor |
| all lr1e-3 | 0.0029 | 15.6/5.2 | 0.980 | 0.00 | 6 | 0.004 / 6.56 | | floor |
| last4 lr1e-3 | 0.00084 | 0.0009/0.011 | 0.999 | 1.00 | 45 | 0.855 / 0.062 | 0.991 / 0.009 | 0.040 / 0.018 |
| last4 lr3e-3 | 0.0036 | 1.25/1.65 | 0.696 | 0.643 | 74 | 0.248 / 1.70 | 0.506 / 1.75 | 1.99 / 2.65 |
| gamma lr1e-3 | – | ~10 | 0.66 | ~0.01 | ~27 | 0.015 / 4.54 | | |
| gamma lr1e-2 | 0.0076 | 10.3/4.2 | 0.46 | 0.032 | 5 | 0.022 / 4.97 | | |

Reading:
- Faithful JiT (28 steps, because the forget set is 3,500 images rather than the paper's 100) collapses the model to a near-constant output for every scope at the paper/code lr. Even at lr 1e-4 on all params it destroys 79% of argmaxes.
- JiT is NOT selective. At matched clean CE, it changes argmax on the proxies (cm agree 0.25 at last4 lr3e-3) while temperature scaling keeps 100% agreement. JiT's KL on the proxies is only slightly lower than the control's (1.70 vs 1.99 on cm), so it would lose heavily on the retain-accuracy factor.
- last4 at lr 1e-3 barely moves anything. The usable band, if any, is last4 between lr 1e-3 and 3e-3, or interpolation toward lr3e-3.
- Decision: JiT is deprioritised. Two interpolated last4-lr3e-3 candidates are exported for official calibration.

Exported JiT candidates (BX_1). These are interpolations original→last4 lr3e-3:
- `BJ_last4_lr0.003_s0.5_N25_a0.25`: clean CE 0.0016, clean agree 1.0, cm agree 0.832, cm kl 0.083, mild kl 0.018.
- `BJ_last4_lr0.003_s0.5_N25_a0.5`: clean CE 0.031, clean agree 0.991, cm agree **0.64**, cm kl 0.40, mild kl 0.185. Its TEMPCTRL has cm kl 0.73, but argmax is kept.

Conclusion: JiT is anti-selective here. The pseudo-control (CutMix) loses argmax much faster than the clean forget images gain CE. Neither export is recommended.

## C-consensus distillation (BC_1: 3 epochs, batch 64, AdamW; anchor λ=1)
Teacher statistics (original, K=8 mild + 8 strong views, seeds 1000k+7):
- prob-avg@T2: pl-prob 0.817, argmax≠pl 0.3%, KL(q‖p0) 0.66.
- logit-avg@T2: pl-prob 0.946, KL 0.097.
- Original mild views: CE vs clean pl 0.076/0.567, agree 0.979.
- Original strong views: CE vs clean pl 0.897/2.213, agree 0.803.
- QM[mix] target CE (clean pl): mean 0.489, sd 1.686, q05/50/95 = 0 / 0 / 3.58.

| run (scope, lr) | drift | clean CE mean/sd | clean conf | clean agree | ncls | mild agr/kl | strong agr/kl | diag agr/kl | cm agr/kl | TEMPCTRL cm kl / diag kl |
|---|---|---|---|---|---|---|---|---|---|---|
| prob β1 last4 3e-5 | .0121 | 0.0103/0.050 | 0.991 | 0.9994 | 45 | .969/.108 | .913/.120 | .978/.114 | .878/.080 | 0.44/0.34 |
| logit β1 last4 3e-5 | .0086 | 0.0054/0.064 | 0.997 | 0.998 | 45 | .980/.022 | .960/.021 | .989/.023 | .953/.010 | 0.30/0.20 |
| prob β.5 last4 3e-5 | .0104 | 0.0025/0.012 | 0.998 | 1.0 | 45 | .981/.043 | .939/.053 | .989/.049 | .920/.031 | 0.16/0.09 |
| prob β1 all 1e-5 | .0085 | 0.0102/0.058 | 0.991 | 0.9994 | 45 | .962/.112 | .910/.109 | .981/.093 | .843/.102 | 0.44/0.33 |
| prob β1 last4, anchor=mild | .0122 | 0.0063/0.028 | 0.994 | 0.9997 | 45 | .987/.072 | .905/.136 | .980/.122 | .800/.183 | 0.33/0.23 |
| prob β1 last4, forget input aug | .0095 | 0.0023/0.012 | 0.998 | 1.0 | 45 | .988/.040 | .940/.052 | .993/.035 | .908/.044 | 0.15/0.08 |
| prob β1 gamma 1e-3 | .0070 | 0.0118/0.056 | 0.990 | 0.9994 | 45 | .965/.141 | .883/.184 | .968/.173 | .810/.134 | 0.47/0.37 |
| QM[mix] last4 3e-5 | .0114 | 0.0484/0.269 | 0.983 | 0.977 | 64 | .969/.054 | .934/.044 | .977/.055 | .925/.030 | 0.85/0.84 |

Reading:
- The T=2 consensus targets are nearly as confident as the original once read at T=1, so the clean CE barely rises (≤0.01).
- They are still more selective than temperature: at matched clean CE, cm kl is 0.03–0.18, against 0.15–0.47 for the control.
- QM is the only variant with a heavy tail (2.3% argmax change, ncls 64). It has the best selectivity (cm kl 0.030 against 0.85 for the control), but it undershoots its target (CE 0.048 against 0.49), so the next step is stronger optimisation.
- A mild-view anchor hurts the pseudo-control more than a CutMix anchor does (cm agree 0.80 against 0.88).
