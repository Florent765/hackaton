"""N01: expected-mIoU decoding (Nowozin, CVPR 2014 style) on saved probabilities. No retraining.
For an image with calibrated cell probabilities q (N cells), predicting the top-k cells positive gives (ratio-of-
expectations approximation):  E[mIoU](k) = ( sum_S q / (k + sum_notS q) + sum_notS (1-q) / (N-k + sum_S (1-q)) ) / 2
We choose k* = argmax_k per image. Calibration q = sigmoid(logit(p) / T + b) with (T, b) per method chosen
  - explor : on the OOF predictions of the evaluated images (exploratory)
  - confirm: on the inner-fold predictions of each outer training set, applied to the outer fold (confirmatory)
Baseline in the same file: REF two-threshold policies on the same probabilities (paired).
usage: python eval_n01.py <exp> [seeds...]"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
from compare import boot_ci  # noqa: E402

DEV = "cuda"
TS = [0.5, 0.75, 1.0, 1.5, 2.0]
BS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def kstar_masks(q):
    """q [B,H,W] -> binary masks maximising expected two-class IoU per image."""
    B = q.shape[0]; f = q.flatten(1); N = f.shape[1]
    s, order = torch.sort(f, dim=1, descending=True)
    cs = torch.cumsum(s, 1)                                       # sum of top-k q, k = 1..N
    tot = cs[:, -1:]
    k = torch.arange(1, N + 1, device=q.device).float()[None]
    i1 = cs; u1 = k + (tot - cs)
    i0 = (N - k) - (tot - cs); u0 = (N - k) + (k - cs)
    e = (i1 / u1.clamp(min=1e-6) + i0 / u0.clamp(min=1e-6)) / 2
    # k = 0 option (all clean): IoU1 = 1 if no positive mass expected (approx: 0 unless tot ~ 0), IoU0 = (N-tot)/N
    e0 = ((N - tot) / N + (tot < 0.5).float())[:, 0] / 2
    best = e.argmax(1); use0 = e0 > e.gather(1, best[:, None])[:, 0]
    kk = torch.where(use0, torch.zeros_like(best), best + 1)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N, device=q.device).expand(B, N))
    return (rank < kk[:, None]).view_as(q)


def calib(p, T, b):
    lg = torch.logit(p.clamp(1e-4, 1 - 1e-4))
    return torch.sigmoid(lg / T + b)


def best_tb(p, gt):
    sc = {(T, b): float(R.miou_batch(kstar_masks(calib(p, T, b)), gt).mean()) for T in TS for b in BS}
    return max(sc, key=sc.get)


def main():
    exp = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or None
    d = R.OUT_ROOT / exp
    recs = [torch.load(p, weights_only=False) for p in sorted(d.glob("seed*.pt"))
            if seeds is None or int(p.stem[4:]) in seeds]
    S = R.summarize(recs)
    allv = {"ref_confirm": [], "ref_explor": [], "n01_confirm": [], "n01_explor": []}; meth = []
    for method in recs[0]["methods"]:
        ms = [r["methods"][method] for r in recs]
        p = torch.stack([m["oof"].float() for m in ms]).mean(0).to(DEV)
        gt = (ms[0]["masks"] if "masks" in ms[0] else R.P.load_cache(method)["masks"]).float().to(DEV)
        fold = ms[0]["fold"].to(DEV)
        tb = best_tb(p, gt); ex = R.miou_batch(kstar_masks(calib(p, *tb)), gt)
        conf = torch.zeros(len(gt), device=DEV); tbs = {}
        for k in range(R.P.K):
            idx = ms[0]["inner"][k]["idx"].to(DEV)
            ip = torch.stack([m["inner"][k]["probs"].float() for m in ms]).mean(0).to(DEV)
            t = best_tb(ip, gt[idx]); tbs[k] = t
            va = (fold == k).nonzero()[:, 0]
            conf[va] = R.miou_batch(kstar_masks(calib(p[va], *t)), gt[va])
        allv["n01_explor"] += ex.tolist(); allv["n01_confirm"] += conf.tolist()
        allv["ref_explor"] += S["per_image"][method]["explor"].tolist(); allv["ref_confirm"] += S["per_image"][method]["confirm"].tolist()
        meth += [method] * len(gt)
        print(f"{method}: explor T,b={tb} n01 {float(ex.mean()):.4f} vs ref {S['by_method'][method]['explor']:.4f} | "
              f"confirm n01 {float(conf.mean()):.4f} vs ref {S['by_method'][method]['confirm']:.4f} (inner T,b {tbs})")
    meth = np.array(meth)
    for pol in ("explor", "confirm"):
        a = np.array(allv[f"n01_{pol}"]); b = np.array(allv[f"ref_{pol}"]); dlt = a - b
        lo, hi = boot_ci(dlt, meth)
        print(f"{pol}: ref {b.mean():.4f} n01 {a.mean():.4f} delta {dlt.mean():+.4f} CI95 [{lo:+.4f},{hi:+.4f}] "
              f"W/L {(dlt > 1e-9).sum()}/{(dlt < -1e-9).sum()}")
    print("---\nval_miou: 0.000000\nN01 DONE")


if __name__ == "__main__":
    main()
