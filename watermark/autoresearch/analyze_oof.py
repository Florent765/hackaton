"""MEGA_PROMPT step A: error analysis of saved out-of-fold probabilities (train.py --save-oof).
Light numpy, OK on a login node.  python analyze_oof.py <oof.pt>
Reports mIoU by method x full_watermark x coverage bin, and oracles:
  mode+const   all-1 if coverage > 0.5 else all-0 (perfect mode, constant mask)
  thr/image    best threshold per image from a grid (upper bound for adaptive thresholding)
  mode-known   our probabilities, but threshold chosen separately for the two modes (full_watermark known)"""
import sys

import numpy as np
import torch

import prepare as P

oof = torch.load(sys.argv[1], weights_only=False)
GRID_T = np.round(np.arange(0.05, 0.96, 0.05), 2)


def miou(p, g):
    s = 0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


rows = []
for m in P.METHODS:
    c = P.load_cache(m); o = oof[m]
    assert o["names"] == c["names"]
    pr = o["probs"].float().numpy(); gt = c["masks"].numpy(); fw = c["full_watermark"].numpy()
    bacc = c["feats"]["scalars"][:, 1].numpy()
    for i in range(len(gt)):
        g = gt[i]; cov = g.mean()
        rows.append(dict(method=m, fw=bool(fw[i]), cov=cov, bacc=float(bacc[i]),
                         ours=miou((pr[i] >= o["thr"]).astype(np.uint8), g),
                         const=miou(np.full_like(g, 1 if cov > 0.5 else 0), g),
                         thr_img=max(miou((pr[i] >= t).astype(np.uint8), g) for t in GRID_T),
                         pred_cov=float((pr[i] >= o["thr"]).mean())))


def summ(sel, label):
    r = [x for x in rows if sel(x)]
    if not r:
        return
    f = lambda k: np.mean([x[k] for x in r])
    print(f"{label:38s} n={len(r):3d}  ours {f('ours'):.4f}  mode+const {f('const'):.4f}  thr/image {f('thr_img'):.4f}"
          f"  cov {f('cov'):.3f} predcov {f('pred_cov'):.3f} bacc {f('bacc'):.2f}")


summ(lambda x: True, "ALL")
for m in P.METHODS:
    summ(lambda x, m=m: x["method"] == m, m)
    for fw in (False, True):
        summ(lambda x, m=m, fw=fw: x["method"] == m and x["fw"] == fw, f"  {m} full_wm={fw}")
for lo, hi in ((0, .05), (.05, .5), (.5, .95), (.95, 1.01)):
    summ(lambda x, lo=lo, hi=hi: lo <= x["cov"] < hi, f"coverage [{lo},{hi})")
# mode errors: predicted coverage on the wrong side of 0.5
wrong = [x for x in rows if (x["pred_cov"] > 0.5) != (x["cov"] > 0.5)]
print(f"mode errors (pred cov vs true cov across 0.5): {len(wrong)}/{len(rows)}; their mean ours {np.mean([x['ours'] for x in wrong]) if wrong else float('nan'):.4f}")
# loss decomposition: how much of (1 - ours) comes from each coverage bin
tot = sum(1 - x["ours"] for x in rows)
for lo, hi in ((0, .05), (.05, .5), (.5, .95), (.95, 1.01)):
    s = sum(1 - x["ours"] for x in rows if lo <= x["cov"] < hi)
    print(f"share of lost mIoU from coverage [{lo},{hi}): {s / tot:.2%}")
