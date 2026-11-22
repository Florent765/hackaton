# B1 (Frankenstein stitching) and B2 (identity amnesia): results

All numbers below are local diagnostics from `ul_lib.full_diag` on the 3,500 forget images. Pseudo-labels are the original model's argmax, and KL is KL(orig‖sub) at T=2. These are not official scores.
Columns: cCE = clean pseudo-label CE (mean / std / q95), cConf = clean confidence, KLc/m/s/d = KL on clean / mild / strong / diag views, chg = argmax change rate.
For reference, the original model scores cCE 0.0003 / 0.002 / 0.001 and cConf 0.9997. Its strong-view CE is 0.117 and its strong-view confidence is 0.910 with these lib views.
Logs: `~/ul_queue/{B1_p1,B1_p2,B2_p1,B2_p2}/run.log`. Summariser: `~/ul/branches/b12_sum.py <log>`.

## B1: stitching (script `branches/B1_stitch.py`, cut = block 6, junction = block 6 re-perturbed and then distilled on strong views only)
- Descendant A = θ0 plus Gaussian noise on all block matrices, with σ = σ_rel × std(W). It is healed with KL(orig‖·) on strong views.
- Descendant B is built the same way with a different noise seed and is healed on mild views.
- Each stitch takes the early blocks [0,6) and patch embedding from one model, the junction (block 6) re-noised with seed 33 and distilled for jsteps, and blocks [7,12) plus norm and head from the other.
- Controls: stOO (original-to-original, same junction treatment) and stAA (same lineage, no crossing).

| run | variant | steps | cCE mean/std/q95 | cConf | KLc | KLm | KLs | KLd | chg clean/strong/diag |
|---|---|---|---|---|---|---|---|---|---|
| p1 σ=0.5 | A (heal strong) | 500 | 0.0015/0.017/0.003 | 0.9987 | 0.007 | 0.022 | 0.059 | 0.020 | 0.000/0.082/0.007 |
| p1 | stAB + junction | 300 | 0.0018/0.027/0.003 | 0.9986 | 0.008 | 0.024 | 0.063 | 0.021 | 0.001/0.086/0.009 |
| p1 | stAA + junction (ctrl) | 300 | 0.0013/0.013/0.003 | 0.9988 | 0.007 | 0.021 | 0.057 | 0.020 | 0.000/0.080/0.008 |
| p1 | stOO + junction (ctrl) | 300 | 0.0003/0.003/0.001 | 0.9997 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000/0.006/0.001 |
| p2 σ=2 | A noised, before heal | 0 | 6.41/2.23/9.92 | 0.452 | 4.74 | 4.61 | 3.93 | 4.59 | 0.97/0.97/0.98 |
| p2 | A (heal strong) | 800 | 0.342/0.719/1.80 | 0.864 | 0.557 | 0.642 | 0.588 | 0.634 | 0.120/0.264/0.171 |
| p2 | B (heal mild) | 800 | 0.032/0.220/0.081 | 0.984 | 0.106 | 0.131 | 0.879 | 0.220 | 0.007/0.357/0.037 |
| p2 | stAB, no junction | 0 | 0.747/1.21/3.32 | 0.787 | 0.977 | 1.034 | 1.142 | 1.109 | 0.245/0.472/0.322 |
| p2 | stAB + junction | 400 | 1.040/1.39/3.98 | 0.731 | 1.205 | 1.248 | 1.254 | 1.277 | 0.344/0.503/0.400 |
| p2 | stAA + junction (ctrl) | 400 | 0.796/1.22/3.46 | 0.764 | 1.013 | 1.095 | 0.934 | 1.086 | 0.270/0.384/0.326 |
| p2 | stOO + junction (ctrl) | 400 | 0.0006/0.004/0.002 | 0.9994 | 0.002 | 0.007 | 0.018 | 0.006 | 0.000/0.048/0.006 |
| p2 | stAB α=0.5 | | 0.144/0.495/0.771 | 0.945 | 0.244 | 0.342 | 0.438 | 0.320 | 0.049/0.255/0.099 |

Run times: p1 took 349 s in total (6 models plus α grid). p2 took about 8 min.

**Decision: falsified.**
- At σ=0.5 the descendants stay in the original's basin. Stitching them without any junction is harmless, and clean memorisation is untouched (cConf ≥ 0.998).
- At σ=2, stAB is more damaged than its parents, but the effect is not specific to forgetting:
  - The KL is roughly uniform across clean, mild, strong and diag views (KLc/KLs = 0.96 for stAB, 1.08 for stAA).
  - stAA, which uses the same lineage with no crossing, gives the same profile.
- The junction is not what breaks memorisation. Retraining the same junction inside the original (stOO) brings back cConf 0.9994.
- So "private complicity" between layers shows nothing beyond generic noise-and-heal damage.
- Side finding: distilling on mild views restores clean memorisation (B: cCE 0.03), while distilling on strong views does not (A: cCE 0.34). The teacher's mild-view logits re-teach the memorised clean behaviour.

## B2: identity amnesia (script `branches/B2_identity.py`)
- Training: blocks 6–11 and the final norm, lr 3e-5, batch 64, 600 steps.
- Anchor: KL(orig‖model) T=2 on strong views only.
- Adversaries: linear probes (LayerNorm + Linear 768→3500) on the CLS token after block 7, after block 9 and after the final norm. They are warm-started on 6 views of the original and updated online. The model minimises CE between each adversary's softmax and the uniform distribution, on both strong and mild views.
- Control: λ=0, which uses the same anchor, the same adversaries and the same number of steps.
- Decisive test: after training, NEW probes (linear, and an MLP 768→1024→3500) are fitted on 6 fresh view seeds and tested on clean, diag and unseen strong views.

| variant | cCE mean/std/q95 | cConf | KLc | KLm | KLs | KLd | chg clean/strong/diag | fresh-probe instance-id acc (clean, block 7/9/norm, linear) |
|---|---|---|---|---|---|---|---|---|
| original | 0.0003/0.002/0.001 | 0.9997 | 0 | 0 | 0 | 0 | 0/0/0 | 1.00 / 1.00 / 1.00 |
| adv λ=1, α=1 | 0.186/0.417/0.937 | 0.894 | 0.507 | 0.868 | 0.359 | 0.398 | 0.053/0.174/0.078 | 1.00 / 1.00 / 1.00 (norm MLP on strong views: 0.955) |
| adv λ=1, α=0.5 | 0.012/0.111/0.019 | 0.994 | 0.029 | 0.087 | 0.078 | 0.045 | 0.004/0.089/0.022 | not measured |
| ctrl λ=0 | 0.0004/0.003/0.001 | 0.9996 | 0.0004 | 0.0015 | 0.0058 | 0.0014 | 0.000/0.031/0.003 | 1.00 / 1.00 / 1.00 |
| p2: adv λ=1, α=0.75 | 0.065/0.311/0.263 | 0.968 | 0.158 | 0.341 | 0.167 | 0.150 | 0.021/0.129/0.047 | |
| p2: adv λ=3, α=1 | 0.533/0.655/1.910 | 0.722 | 1.137 | 1.550 | 0.809 | 0.955 | 0.155/0.267/0.182 | 1.00 / 1.00 / 1.00 |
| p2: adv λ=3, α=0.75 | 0.216/0.546/1.212 | 0.904 | 0.437 | 0.723 | 0.370 | 0.393 | 0.074/0.198/0.103 | |
| p2: adv λ=3, α=0.5 | 0.053/0.335/0.154 | 0.980 | 0.092 | 0.202 | 0.158 | 0.119 | 0.015/0.127/0.048 | |

| p3 ctrl: frozen RANDOM probes λ=1, α=1 | 0.639/0.710/2.117 | 0.677 | 1.243 | 1.487 | 0.929 | 1.151 | 0.193/0.301/0.218 | n/a |
| p3 ctrl: random probes λ=1, α=0.75 | 0.255/0.543/1.418 | 0.881 | 0.497 | 0.703 | 0.424 | 0.464 | 0.087/0.218/0.123 | |
| p3 ctrl: random probes λ=3, α=1 | 3.70/1.24/6.07 | 0.094 | 3.79 | 3.67 | 3.16 | 3.67 | 0.86/0.86/0.86 (collapse) | |

p3 is the non-identity control: the same pipeline, but the probes are frozen and random (weights ×30), so the model flattens arbitrary readouts and not instance identity. At matched strong-view KL it gives the same clean/mild-selective profile:
- random λ=1, α=0.75: KLs 0.42, KLc 0.50, KLm 0.70, cCE 0.25
- identity λ=1, α=1: KLs 0.36, KLc 0.51, KLm 0.87, cCE 0.19
- identity λ=3, α=0.75: KLs 0.37, KLc 0.44, KLm 0.72, cCE 0.22

So the movement comes from "strong-only KL anchor plus a feature-flattening push on mild and strong views", not from the identity objective. **Final B2 decision: falsified as a mechanism.** The models remain usable as generic movers.

p2 (403 s) reproduced p1 λ=1 bit-for-bit. It exported B2_p2_adv_l{1.0,3.0}_a{0.5,0.75,1} to `~/ul_out`.

Run time: 283 s for the whole run (original, adv and control, each with probes).

**Decision: falsified as identity amnesia.**
- The online adversaries were never fooled: their training accuracy stayed at about 0.96.
- Fresh probes re-identify 100% of the clean images at every depth.
- Instance identity can be decoded at about 100% from the original features. Any feature extractor would do the same, so it is not a memorisation signature.
- The adversarial term does move the model in a way that is somewhat clean/mild-selective: KLc/KLs = 1.41, against about 1.0 for noise-and-heal. It acts as a structured perturbation, not as "losing image #247". It may still be worth an official score as a mover. The α curve is very non-linear, since α=0.5 is almost the original.

## Candidates for official scoring (already exported to `/p/home/jusers/jacquet2/jureca/ul_out/`)
1. `B2_p2_adv_l1.0_a1`: cCE 0.186, cConf 0.894, clean argmax change 5.3%, KLs 0.36, KLd 0.40
2. `B2_p2_adv_l1.0_a0.75`: cCE 0.065, cConf 0.968, change 2.1%, KLs 0.17, KLd 0.15 (the milder version)
3. `B2_p2_adv_l3.0_a0.75`: cCE 0.216, cConf 0.904, change 7.4%, KLs 0.37, KLd 0.39 (heavier clean tail, q95 1.21)

Also exported, but lower priority: `B2_p2_adv_l1.0_a0.5`, `B2_p2_adv_l3.0_a0.5`, `B2_p2_adv_l3.0_a1`. No B1 model was exported, since that branch is falsified.
