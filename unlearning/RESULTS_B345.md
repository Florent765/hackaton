# Branches B3 / B4 / B5 (diagnostics only, not the official score)

All numbers are from `L.full_diag` on the 3,500 forget images (seed-0 views), against the original's logits on the same
view. Pseudo-labels are the original's argmax, never true labels. `kl` = KL(orig||sub) at T=2; `ce` = pseudo-label CE
(mean / sd, q05/50/95); `chg` = argmax change. Original: clean ce 0.0003 (sd 0.0024), conf 0.9997; mild ce 0.017;
strong ce 0.117, conf 0.910; diag ce 0.011, conf 0.991.
Scripts: `unlearning/branches/B_common.py`, `B3_bayes.py`, `B4_modes.py`, `B5_evo.py`, `B_export.py`.

## B4: change the solution, keep competence (re-init late MLPs, re-fit by distillation on views)

Each run took 1.5–2.5 min of training (800–1200 AdamW steps, batch 96, KL at T=2 against the original) plus about 1 min per alpha evaluation.

| run | clean kl | clean ce mean/sd | clean conf | mild kl | strong kl | diag kl | diag chg |
|---|---|---|---|---|---|---|---|
| perm control (b10-11, weights moved by a relative L2 distance of 1.39) | 0.0 | 0.0003/0.0024 | 0.9997 | 0.0 | 0.0 | 0.0 | 0.0003 |
| re-init b10-11, no re-fit (500 held-out images)* | 1.14 | 0.67/1.10 | 0.763 | (1.37) | (1.81) | (1.21) | (0.26) |
| mlp (strong+mild) b10-11, 800 steps, alpha 1 | 0.018 | 0.0021/0.0275 | 0.998 | 0.035 | 0.154 | 0.052 | 0.009 |
| mlps (STRONG only) b10-11, 1200 steps, alpha 1 | 0.023 | 0.0029/0.0241 | 0.997 | 0.053 | 0.106 | 0.050 | 0.010 |
| mlps b10-11, alpha 0.5 | 0.349 | 0.088/0.262 | 0.942 | 0.477 | 0.519 | 0.463 | 0.079 |
| mlpc control (strong+mild+CLEAN) b10-11, alpha 1 | 0.008 | 0.0004/0.0027 | 0.9996 | 0.028 | 0.146 | 0.040 | 0.008 |
| mlps STRONG only b6-11, 1200 steps, alpha 1 | 0.116 | 0.034/0.146 | 0.976 | 0.166 | 0.249 | 0.169 | 0.030 |
| mlps b6-11, alpha 0.5 | 1.08 | 0.93/1.33 | 0.762 | 1.23 | 1.02 | 1.17 | 0.386 |

*The mild, strong and diag numbers in the re-init row come from a subset `idx` call, where the view seeds do not match the teacher cache, so they are biased upward. Only the clean column is valid. This bug was found in B3 and fixed in B3/B5. Every other row uses the full index, so its views match.

Findings:
- The permutation control is function-identical, so the diagnostics are sound.
- Re-fitting on STRONG views only, without ever seeing a clean forget image, brings back clean-image memorisation almost completely (ce 0.0029, conf 0.997). The clean-image control adds only a little on top (0.0004).
- So the "image-specific shortcut is reachable only through clean images" hypothesis is **falsified** for late MLPs. The memorisation is either readable from the upstream features (blocks 0–9 or 0–5) or re-learned from the views.
- With 6 blocks re-initialised, clean ce rises to 0.034. Mild, strong and diag KL rise by the same or more (0.17 / 0.25 / 0.17), so the clean images are not singled out.
- alpha 0.5 sits on the loss barrier between the two solutions. It damages every view about equally, which is generic damage rather than forgetting.

Decision: **falsified** as a targeted mechanism. Exported for official scoring anyway, at the main session's request:
`B4_mlps_b1011_a05`, `B4_mlps_b1011_a10`, `B4_mlps_b6to11_a10`.

## B3: Bayesian division (Laplace / influence-function removal step on P = blocks.11 + norm + head, 7.2M params)

Removal step: theta_R = theta0 + H_tot^{-1} grad_P NLL_F(theta0), where NLL_F is the clean forget images under HARD pseudo-labels.
- With "soft own predictions" as targets the gradient is identically zero, because sum_y p grad log p = grad sum_y p = 0. Measured: |g_soft| = 2.5e-4 against |g_hard| = 9.1, so it is numerical noise. Hard pseudo-labels avoid this.
- H_tot proxy is the true Fisher (labels sampled from the model) on 512 mild + strong views of the FIT images:
  (a) diagonal, (b) K-FAC per linear layer (diagonal for LN and LayerScale), (c) SWAG-like covariance from 9 SGD snapshots.
- Each direction is scaled so that KL(orig||sub) on 400 held-out mild views equals the target, then evaluated at alpha 0.25 / 0.5 / 1.

Cosines between the update directions (first test). In the full P space: grad|diag 0.154, grad|kfac 0.143, grad|swag 0.03, **diag|kfac 0.037, diag|swag 0.001, kfac|swag 0.000**, random about 0. On the head only, diag|kfac is 0.15.
-> The posterior approximations do **not** agree on the direction. The curvature-specific part of the update is hypothesis-driven noise; the only shared component is the raw gradient.

Bug fixed before these numbers: in the first B3 run (B3_p1), every scale came out degenerate. The cause is a seed mismatch when `L.logits(..., idx=subset)` is compared with `tc[...][subset]`: the views are seeded by the offset inside the subset, not the global index. The fixed run checks that KL is 0.0 at delta=0.
**This affects anyone who calls `L.full_diag(..., idx=...)` with a subset on non-clean views.**

Time: 4 min per run (B3_p2). Results at a target of KL_mild = 0.05 (full 3,500 images):

| direction (|delta|/|theta_P|) | alpha | clean kl | clean ce mean/sd (q95) | clean chg | mild kl | strong kl | diag kl | strong chg |
|---|---|---|---|---|---|---|---|---|
| diag (0.028) | 1 | 0.053 | 0.097/1.57 (0.005) | 0.014 | 0.061 | 0.070 | 0.046 | 0.034 |
| kfac (0.275) | 0.5 | 0.044 | 0.109/1.35 (0.002) | 0.010 | 0.025 | 0.009 | 0.022 | 0.018 |
| kfac (0.275) | 1 | 0.090 | 0.205/2.01 (0.009) | 0.020 | 0.063 | 0.038 | 0.060 | 0.037 |
| swag (0.015) | 1 | 0.026 | 0.025/0.31 | 0.007 | 0.056 | 0.117 | 0.044 | 0.102 |
| grad (0.012) | 1 | 0.056 | 0.100/1.06 | 0.017 | 0.061 | 0.073 | 0.052 | 0.082 |
| random control (0.81) | 1 | 0.024 | 0.0055/0.06 | 0.001 | 0.053 | 0.122 | 0.050 | 0.117 |

The principled diagonal Newton scale (assuming N_tot = 50k, prior precision 1) gives |d| = 4.17, KL_mild = 0.116.

Reading:
- Unlike the random control and B4, the K-FAC removal step moves clean images more than strong views (clean/strong KL ratio 2.4, against 0.2 for random and 0.36 for SWAG).
- The change is a heavy tail: median clean CE stays at 0, the sd jumps to 2.0, and about 2% of argmaxes flip. It is not a uniform loss of confidence.
- That is qualitatively gold-like (heavier tail), but far too small against the anchor of gold agreement around 0.85–0.95.

Larger steps (B3_p3, 6 min):

| direction, target | alpha | clean kl | clean ce mean/sd (q50, q95) | clean conf | clean chg | mild kl | strong kl | diag kl | diag chg |
|---|---|---|---|---|---|---|---|---|---|
| kfac kl0.2 (rel 0.55) | 0.5 | 0.090 | 0.206/2.01 (0.0, 0.009) | 0.994 | 0.020 | 0.064 | 0.038 | 0.060 | 0.015 |
| kfac kl0.2 | 1 | 0.231 | 0.472/2.84 (0.0, 0.93) | 0.988 | 0.052 | 0.200 | 0.169 | 0.169 | 0.043 |
| diag kl0.2 (rel 0.039) | 1 | 0.167 | 0.318/2.73 (0.0, 0.13) | 0.992 | 0.038 | 0.187 | 0.211 | 0.139 | 0.036 |
| kfac kl0.5 (rel 0.84) | 0.5 | 0.153 | 0.326/2.45 (0.0, 0.089) | 0.991 | 0.035 | 0.123 | 0.092 | 0.109 | 0.028 |
| kfac kl0.5 | 1 | 0.508 | 0.982/3.89 (0.0001, 7.09) | 0.976 | 0.107 | 0.484 | 0.481 | 0.396 | 0.103 |
| diag kl0.5 (rel 0.053) | 1 | 0.388 | 0.752/4.37 (0.0, 3.08) | 0.987 | 0.073 | 0.420 | 0.472 | 0.320 | 0.066 |

- K-FAC stays more selective than the diagonal: at matched clean KL it moves strong views less. The selectivity fades as the step grows, and at kl0.5 with alpha 1 clean and strong are about equal.
- The removal step is always heavy-tailed: a growing minority of images flips with very high CE while the median image stays at conf about 1. Mean conf never drops below 0.976.
- This is not the gold anchor (agreement 0.85–0.95 *with lowered confidence on most images*), so it is at best one component of a solution.
- The posterior approximations disagree on direction (cos 0.00–0.04), so the "Bayesian" content beyond the gradient is not identifiable from this data.

Decision B3: **inconclusive, weakly promising as a component.** K-FAC removal gives gold-like heavy tails and some clean-over-strong selectivity at small steps. The directions are not robust across approximations.
Exported: `B3_kfac_kl02_a10`, `B3_kfac_kl05_a05`.

## B5: evolving population of amnesic models (NSGA-II-lite, population 12, 8 generations, 5 min total)

Genes (27):
- per-block log-scales of the LayerScale gammas (ls1 and ls2, 12 blocks each);
- head temperature (log-scale of head W and b);
- a negative bias shift on the 45 pseudo-label classes;
- interpolation strength (0..2) along the B3 K-FAC kl0.2 removal step.

No distillation-step gene: too costly for the time budget.

Objectives, all minimised and computed on 600 FIT images:
- f1 = KL_T2(orig on strong views, 2 seeds averaged || model on clean);
- f2 = KL(orig||model) on mild views;
- f3 = KL(orig||model) on strong views with a different seed.

Held-out check uses 500 HELD images on the reserved diag views. They are never used in selection.

Pareto front (12 points, from the original at f1 = 1.67 down to f1 = 0.72):

| f1 | f2 mild | f3 strong | held f1 | held KL diag | held clean ce mean/sd | held clean conf | temp | fbias | B3 step |
|---|---|---|---|---|---|---|---|---|---|
| 1.673 (orig) | 0 | 0 | 1.098 | 0 | 0.0002/0.001 | 0.9998 | 1.0 | 0 | 0 |
| 1.276 | 0.022 | 0.043 | 0.876 | 0.014 | 0.0009/0.004 | 0.9991 | 0.86 | 0 | 0 |
| 0.991 | 0.078 | 0.136 | 0.711 | 0.075 | 0.0030/0.012 | 0.9971 | 0.75 | 0 | 0 |
| 0.877 | 0.131 | 0.172 | 0.670 | 0.119 | 0.0078/0.034 | 0.9928 | 0.73 | -0.07 | 0 |
| 0.822 | 0.220 | 0.328 | 0.679 | 0.207 | 0.0218/0.145 | 0.9878 | 0.72 | 0 | 0 |
| 0.786 | 0.283 | 0.441 | 0.652 | 0.281 | 0.0285/0.254 | 0.9852 | 0.66 | -0.48 | 0.04 |
| 0.721 | 0.345 | 0.448 | 0.647 | 0.308 | 0.0223/0.134 | 0.9853 | 0.61 | 0 | 0 |

Held-out generalisation: rank correlation 0.85 between fit f1 and held f1, and 0.99 between fit f2 and held KL on diag views. The proxies transfer across images and view families.

Reading:
- Evolution pushed the head temperature down (soften all logits, 0.6–0.86) and made mixed LayerScale tweaks.
- It almost always switched off the B3 removal step (alpha about 0) and the forget-class bias shift.
- Along the front, KL on strong views is at least KL on mild views, and the full-diagnostic check of the picks confirms that clean images are never moved more than generic views.
  - pick_b0.3 at alpha 1: clean kl 0.209, mild 0.273, strong 0.427, diag 0.318; clean ce 0.032/0.236, conf 0.982, argmax change 0.6%.
  - pick_b0.1 at alpha 1: clean kl 0.050, mild 0.075, strong 0.132, diag 0.088.
- The front is essentially a global-softening tradeoff. Pseudo-label agreement on clean stays at 99.4% or more.

Decision B5: **falsified as a selective-forgetting route with these genes.** The selection machinery is sound (held-out transfer) but finds only global calibration. Exported anyway: `B5_pick_b0.1`, `B5_pick_b0.3`.

## Summary / candidates for official scoring

Ranked on the diagnostics, with no score claims:
1. `B3_kfac_kl02_a10`: the most clean-over-generic selectivity, with a heavy-tailed clean loss.
2. `B3_kfac_kl05_a05`.
3. `B5_pick_b0.3`: a global-softening reference.

Also exported: `B4_mlps_b1011_a05`, `B4_mlps_b1011_a10`, `B4_mlps_b6to11_a10`, `B5_pick_b0.1`.
All exports are in `/p/home/jusers/jacquet2/jureca/ul_out/`. Saved raw state dicts are in `/p/project1/training2666/jacquet2/ul/b345/`.
