# Pending autoresearch experiments (prepared, not yet run)

All are single-change variants of the current best `train.py` (e28, occlusion maps, val_miou 0.5041).
Queue at most 1–2 at a time on the team's single job (do not saturate the GPU).

| Id | Change | Why (from MEGA_PROMPT audit) | Cost |
| --- | --- | --- | --- |
| e33_occ_raw | occlusion maps: standardized + raw ×10 channels | per-image standardization may discard the overall strength of the signal | ≈ e28 |
| e34_grid32_fixed | head grid 32 (fix of crashed e18: RGB resized to the head grid) | larger relative receptive field | < e28 |
| e35_cc_clean | drop small predicted components / fill small holes, sizes chosen on OOF | tiny-coverage images score below all-zero (0.486 vs 0.490) | e28 + CPU post-processing |
| e36_seeds5 | 5 seeds instead of 3 | variance reduction; only if nothing else wins | ≈ 1.7× e28 |

Running when prepared: e29_thr2 (two thresholds per mode), e31_foldseed1 (robustness re-check of e28 with fold
seed 1 — not a candidate).

Next ideas needing new caches (GPU time, decide before queueing): PixelSeal occlusion on a 16×16 grid (≈4× the
8×8 cost: ~1.5 min train + 5 min test), StegaStamp keep-window maps (AUROC 0.58 in probe_v2).
