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

## BC_2 (6 epochs, stronger lr; T=1 consensus targets added)
T=1 teachers:
- prob1 (mild+strong prob-avg): conf 0.879, CE-pl 0.138/0.134, argmax≠pl 0.4%.
- strong1: conf 0.786, CE-pl 0.272/0.255, argmax≠pl 1.6%.
- QM[strong] target: CE 0.899/2.223, q95 5.88.

| run | drift | clean CE mean/sd (q95) | conf | agree | ncls | mild agr/kl | strong agr/kl (cecl) | diag agr/kl | cm agr/kl | TEMPCTRL cm/diag kl |
|---|---|---|---|---|---|---|---|---|---|---|
| QM[mix] last4 1e-4 | .033 | 0.178/0.600 (1.50) | .948 | .928 | 71 | .941/.141 | .898/.125 (1.21) | .935/.197 | .879/.059 | 1.25/1.45 |
| QM[mix] all 3e-5 | .021 | 0.110/0.398 (0.92) | .958 | .950 | 66 | .946/.113 | .907/.101 (1.17) | .949/.145 | .889/.052 | 1.10/1.21 |
| prob1 last4 1e-4 | .035 | 0.054/0.130 (0.28) | .955 | .997 | 47 | .955/.187 | .869/.241 (1.18) | .940/.319 | .861/.081 | 0.89/0.89 |
| strong1 last4 1e-4 | .037 | 0.119/0.221 (0.59) | .908 | .988 | 53 | .935/.329 | .821/.389 (1.29) | .901/.527 | .828/.124 | 1.13/1.25 |
| QM[mix] last4 aug 1e-4 | .032 | 0.104/0.468 (0.61) | .967 | .961 | 66 | .943/.150 | .863/.175 (0.72!) | .952/.165 | .872/.066 | 1.08/1.18 |
| QM[strong] last4 1e-4 | .036 | 0.258/0.628 (1.77) | .908 | .892 | 81 | .911/.263 | .843/.252 (1.32) | .889/.382 | .849/.098 | 1.38/1.65 |

Reading:
- QM gives the gold-like shape: argmax kept on 89–95% of images, a heavy CE tail (median ≈0, q95 1–1.8), and up to 81 classes. At matched clean CE its pseudo-control KL is about 20× lower than the temperature control's (0.05–0.10 against 1.1–1.4).
- Consensus targets (prob1/strong1) only smooth uniformly, with no tail, and they drift diag/mild more.
- "aug" (training the clean target on strong views) makes the strong views MORE consistent with the clean label (strong cecl 0.72 against 1.11 for the original). That is the opposite of forgetting on altered copies, so it is rejected.
- The cm agreement drops to 0.85–0.89 for every BC run, including near-no-op ones (0.92). CutMix argmax is fragile, so the independent mixup proxy `mx` was added from BX_2 on.

## BX_2 exports (with the independent mixup proxy `mx`: λx_i+(1−λ)x_j, λ~U(.6,.8), never used as an anchor)
| export | clean CE mean/sd (q95) | clean agree | ncls | mild kl | diag agr/kl | cm agr/kl | mx agr/kl | TEMPCTRL mx kl |
|---|---|---|---|---|---|---|---|---|
| BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a0.5 | 0.0058/0.044 (0.017) | .9994 | 45 | .025 | .988/.033 | .937/.014 | .941/.028 | 0.30 |
| BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a1.0 | 0.178/0.600 (1.50) | .928 | 71 | .141 | .935/.197 | .879/.059 | .873/.127 | 1.34 |
| BC_qm_bmix_all_cm1_clean_e6_lr3e-5_a1.0 | 0.110/0.398 (0.92) | .950 | 66 | .113 | .949/.145 | .888/.052 | .872/.116 | 1.17 |
| BC_qm_bstrong_last4_cm1_clean_e6_lr1e-4_a1.0 | 0.258/0.628 (1.77) | .892 | 81 | .263 | .889/.382 | .848/.098 | .809/.244 | 1.49 |

- Weight interpolation is very nonlinear. At α=0.5 almost all of the forget effect disappears (CE 0.006), so α ∈ {0.75, 1} is the useful range.
- The mx proxy confirms the CutMix ranking. Note that mx is built FROM forget images (70% dominant), so some argmax change there is expected and possibly desirable. It is not a clean retain proxy.

## BC_3 (QM, longer training / scope / anchor strength)
| run | drift | clean CE mean/sd (q95) | conf | agree | ncls | mild kl | strong agr/kl | diag agr/kl | cm agr/kl | mx agr/kl | TEMPCTRL mx kl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QM[mix] last4 λ1 e10 1e-4 | .040 | 0.205/0.671 (1.85) | .944 | .921 | 74 | .143 | .891/.127 | .931/.209 | .875/.060 | .866/.129 | 1.40 |
| QM[mix] last4 λ3 e10 1e-4 | .039 | 0.124/0.448 (1.15) | .958 | .943 | 69 | .090 | .912/.077 | .951/.130 | .877/.049 | .878/.085 | 1.21 |
| QM[mix] last6 λ1 e8 7e-5 | .032 | 0.191/0.640 (1.65) | .947 | .923 | 73 | .141 | .892/.124 | .929/.204 | .885/.055 | .875/.125 | 1.37 |
| QM[mix] all λ1 e10 3e-5 | .025 | 0.142/0.467 (1.24) | .949 | .933 | 72 | .124 | .902/.112 | .944/.168 | .893/.051 | .871/.122 | 1.26 |
| QM[strong] all λ1 e8 3e-5 | .027 | 0.181/0.455 (1.29) | .920 | .920 | 82 | .236 | .857/.229 | .910/.322 | .849/.096 | .798/.257 | 1.35 |

Reading:
- QM[mix] saturates near clean CE ≈0.2 and agree ≈0.92. The target (CE 0.49) is never reached, because the memorized tail resists.
- Scope (last4/last6/all) gives nearly the same frontier. λ=3 gives a more conservative point on that same frontier.
- QM[strong] costs about 2× more proxy KL for a similar clean CE, so it is dominated.

## BC_4 (G = LayerScale gammas only; last2)
| run | drift | clean CE mean/sd (q95) | conf | agree | ncls | mild kl | strong agr/kl | diag agr/kl | cm agr/kl | mx agr/kl | TEMPCTRL mx kl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QM[mix] gamma e8 1e-3 | .0099 | 0.039/0.176 (0.19) | .975 | .990 | 50 | .125 | .875/.162 | .949/.173 | .798/.116 | .797/.206 | 0.82 |
| QM[mix] gamma e8 3e-3 | .0185 | 0.049/0.222 (0.22) | .973 | .984 | 56 | .143 | .855/.190 | .942/.197 | .755/.166 | .790/.235 | 0.89 |
| QM[mix] last2 e8 1e-4 | .034 | 0.146/0.505 (1.23) | .952 | .938 | 67 | .121 | .901/.094 | .944/.144 | .894/.047 | .884/.096 | 1.27 |

Reading:
- The gamma-only scope (G) is the worst option tested. It produces little forget effect, and 2× more proxy KL with 8–10 points more CutMix/mixup argmax change. Rejected.
- last2 lies on the best frontier with the smallest block scope.

## BX_3 exports
- `BC_qm_bmix_last2_cm1_clean_e8_lr1e-4_a1.0`: clean CE 0.146/0.505, agree .938, ncls 67, cm kl .047, mx kl .096 (mx agree .884).
- `BC_qm_bmix_last4_cm3_clean_e10_lr1e-4_a1.0`: clean CE 0.124/0.448, agree .943, ncls 69, cm kl .049, mx kl .085 (mx agree .878).
- `BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a0.75`: clean CE 0.045/0.215, agree .989, ncls 53, cm kl .033, mx kl .067 (mx agree .907).

## Decision / ranking (local diagnostics only; no official scores seen)
QM[mix] (quantile-matched consensus targets with a CutMix anchor) is the best family found here.

Ranked candidates, from main to conservative:
1. `BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a1.0`: main. Agree .928, CE .18/.60.
2. `BC_qm_bmix_last2_cm1_clean_e8_lr1e-4_a1.0`: smallest scope, on the same frontier.
3. `BC_qm_bmix_last4_cm3_clean_e10_lr1e-4_a1.0`: conservative.
4. `BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a0.75`: very conservative, close to the original.

Other exports, not recommended:
- `BC_qm_bmix_all_cm1_clean_e6_lr3e-5_a1.0` and `BC_qm_bstrong_last4_cm1_clean_e6_lr1e-4_a1.0`: dominated or more drift.
- `BC_qm_bmix_last4_cm1_clean_e6_lr1e-4_a0.5`: near the original.
- `BJ_last4_lr0.003_s0.5_N25_a0.25` / `_a0.5`: JiT, anti-selective.

Open risk: none of these proxies is real retain data. CutMix and mixup are built from forget images, so the true control-set KL is unmeasured. The official score is needed to place the operating point.
