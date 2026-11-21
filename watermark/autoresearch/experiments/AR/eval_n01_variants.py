"""Autoresearch lead 2: N01 decoding variants on saved probabilities (no training). Paired vs standard N01.
  fine   : finer calibration grid (T in 0.4..2.5, b in -1.5..1.5)
  regime : separate (T, b) for high / low global bit accuracy (> 0.75), chosen on inner folds
usage: python eval_n01_variants.py <exp> [seeds...]"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "N01"))
import rlib as R  # noqa: E402
import eval_n01 as N  # noqa: E402
from compare import boot_ci  # noqa: E402

DEV = "cuda"
FINE_T = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5]
FINE_B = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]


def best(p, gt, Ts, Bs):
    sc = {(T, b): float(R.miou_batch(N.kstar_masks(N.calib(p, T, b)), gt).mean()) for T in Ts for b in Bs}
    return max(sc, key=sc.get)


def decode(p, hi, tb_lo, tb_hi):
    m = torch.zeros_like(p, dtype=torch.bool)
    for sel, tb in ((~hi, tb_lo), (hi, tb_hi)):
        if sel.any():
            m[sel] = N.kstar_masks(N.calib(p[sel], *tb))
    return m


exp = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
recs = [torch.load(R.OUT_ROOT / exp / f"seed{s}.pt", weights_only=False) for s in seeds]
res = {"std": [], "fine": [], "regime": []}; meth = []
for method in recs[0]["methods"]:
    ms = [r["methods"][method] for r in recs]
    p = torch.stack([m["oof"].float() for m in ms]).mean(0).to(DEV)
    gt = (ms[0]["masks"] if "masks" in ms[0] else R.P.load_cache(method)["masks"]).float().to(DEV)
    hi = ms[0]["hi"].to(DEV); fold = ms[0]["fold"].to(DEV)
    out = {k: torch.zeros(len(gt), device=DEV) for k in res}
    for k in range(R.P.K):
        idx = ms[0]["inner"][k]["idx"].to(DEV)
        ip = torch.stack([m["inner"][k]["probs"].float() for m in ms]).mean(0).to(DEV)
        va = (fold == k).nonzero()[:, 0]
        t_std = best(ip, gt[idx], N.TS, N.BS); t_fine = best(ip, gt[idx], FINE_T, FINE_B)
        hi_i = hi[idx]
        t_lo = best(ip[~hi_i], gt[idx][~hi_i], N.TS, N.BS) if (~hi_i).any() else t_std
        t_hi = best(ip[hi_i], gt[idx][hi_i], N.TS, N.BS) if hi_i.any() else t_std
        out["std"][va] = R.miou_batch(N.kstar_masks(N.calib(p[va], *t_std)), gt[va])
        out["fine"][va] = R.miou_batch(N.kstar_masks(N.calib(p[va], *t_fine)), gt[va])
        out["regime"][va] = R.miou_batch(decode(p[va], hi[va], t_lo, t_hi), gt[va])
    for k in res:
        res[k] += out[k].tolist()
    meth += [method] * len(gt)
    print(method, {k: round(float(v.mean()), 4) for k, v in out.items()})
meth = np.array(meth); base = np.array(res["std"])
print(f"N01 standard (confirm): {base.mean():.4f}")
for k in ("fine", "regime"):
    d = np.array(res[k]) - base; lo, hi_ = boot_ci(d, meth)
    print(f"{k}: {np.mean(res[k]):.4f} delta vs N01 {d.mean():+.4f} CI95 [{lo:+.4f},{hi_:+.4f}] W/L {(d > 1e-9).sum()}/{(d < -1e-9).sum()}")
print("---\nval_miou: 0.000000\nN01 VARIANTS DONE")
