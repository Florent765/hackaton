"""A02: rectangle prior for the low-coverage regime, decoded with the expected-mIoU criterion (N04 x N01).
Diagnostic (A02_oracle): in the lo regime (bit acc <= 0.75) ~53% of train masks fill > 98% of their bbox and the
GT-bbox oracle scores 0.85-0.89 vs 0.49 for N03. Here, without labels at decision time:
  q = sigmoid(logit(p)/T + b) (calibrated as N01), pooled sums on a 32x32 grid, exhaustive search (integral
  image) of the axis-aligned rectangle R maximising E[mIoU](R) = (E[I1]/E[U1] + E[I0]/E[U0]) / 2 (same ratio of
  expectations as N01's top-k).
Policies (pre-registered, hi images always use N01 top-k):
  rect : lo images -> best rectangle
  sel  : lo images -> rectangle if E[mIoU](rect) >= E[mIoU](top-k) else top-k
Calibration (T, b) per method and policy: explor = on the OOF of the evaluated images; confirm = on the inner-fold
probabilities of each outer training set (outer labels never used). Baselines on the same probabilities: REF
two-threshold policies and N01.
usage: python eval_rect.py <exp> [seeds...]"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/N01")
import rlib as R  # noqa: E402
from compare import boot_ci  # noqa: E402
from eval_n01 import TS, BS, calib  # noqa: E402

DEV = "cuda"
G = 32
PAIRS = torch.tensor([(a, b) for a in range(G) for b in range(a + 1, G + 1)], device=DEV)   # [528,2] (start, end excl)


def kstar(q):
    """top-k expected-mIoU decoding (same as N01, k=0 option included) -> mask, expected score."""
    B = q.shape[0]; f = q.flatten(1); N = f.shape[1]
    s, order = torch.sort(f, dim=1, descending=True)
    cs = torch.cumsum(s, 1); tot = cs[:, -1:]
    k = torch.arange(1, N + 1, device=q.device).float()[None]
    e = (cs / (k + tot - cs).clamp(min=1e-6) + ((N - k) - (tot - cs)) / ((N - k) + (k - cs)).clamp(min=1e-6)) / 2
    e0 = ((N - tot) / N + (tot < 0.5).float())[:, 0] / 2
    best = e.argmax(1); eb = e.gather(1, best[:, None])[:, 0]; use0 = e0 > eb
    kk = torch.where(use0, torch.zeros_like(best), best + 1)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N, device=q.device).expand(B, N))
    return (rank < kk[:, None]).view_as(q), torch.maximum(eb, e0)


def best_rect(q, chunk=32):
    """exhaustive rectangle search on a 32x32 grid of q sums -> mask [B,128,128], expected score [B]."""
    B, H, W = q.shape; N = H * W; c = H // G
    masks, scores = [], []
    for s0 in range(0, B, chunk):
        qq = q[s0:s0 + chunk]; b = len(qq)
        cell = qq.view(b, G, c, G, c).sum((2, 4))
        I = torch.zeros(b, G + 1, G + 1, device=q.device); I[:, 1:, 1:] = cell.cumsum(1).cumsum(2)
        tot = I[:, -1, -1][:, None, None]
        A = I[:, PAIRS[:, 1], :] - I[:, PAIRS[:, 0], :]                          # [b,528,33]
        Sin = A[:, :, PAIRS[:, 1]] - A[:, :, PAIRS[:, 0]]                        # [b,528(y),528(x)]
        ly = (PAIRS[:, 1] - PAIRS[:, 0]).float() * c; lx = ly
        k = ly[:, None] * lx[None, :]                                             # pixels in rect
        e = (Sin / (k + tot - Sin).clamp(min=1e-6) + ((N - k) - (tot - Sin)) / ((N - k) + (k - Sin)).clamp(min=1e-6)) / 2
        flat = e.flatten(1); best = flat.argmax(1); scores.append(flat.gather(1, best[:, None])[:, 0])
        iy = best // len(PAIRS); ix = best % len(PAIRS)
        y0, y1 = PAIRS[iy, 0] * c, PAIRS[iy, 1] * c; x0, x1 = PAIRS[ix, 0] * c, PAIRS[ix, 1] * c
        r = torch.arange(H, device=q.device)
        my = (r[None] >= y0[:, None]) & (r[None] < y1[:, None]); mx = (r[None] >= x0[:, None]) & (r[None] < x1[:, None])
        masks.append(my[:, :, None] & mx[:, None, :])
    return torch.cat(masks), torch.cat(scores)


def policy(p, hi, T, b, pol):
    q = calib(p, T, b); mk, ek = kstar(q)
    if pol == "n01":
        return mk
    lo = (~hi).nonzero()[:, 0]
    if len(lo) == 0:
        return mk
    mr, er = best_rect(q[lo])
    out = mk.clone()
    if pol == "rect":
        out[lo] = mr
    elif pol == "sel":
        use = er >= ek[lo]
        out[lo] = torch.where(use[:, None, None], mr, mk[lo])
    return out


def score(p, gt, hi, T, b, pol):
    return R.miou_batch(policy(p, hi, T, b, pol), gt)


def best_tb(p, gt, hi, pol):
    sc = {(T, b): float(score(p, gt, hi, T, b, pol).mean()) for T in TS for b in BS}
    return max(sc, key=sc.get)


def main():
    exp = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or None
    d = R.OUT_ROOT / exp
    recs = [torch.load(p, weights_only=False) for p in sorted(d.glob("seed*.pt")) if seeds is None or int(p.stem[4:]) in seeds]
    print(exp, "seeds", [r["seed"] for r in recs])
    S = R.summarize(recs)
    pols = ("n01", "rect", "sel")
    allv = {f"{pol}_{m}": [] for pol in pols for m in ("explor", "confirm")}
    allv["ref_confirm"] = []; allv["ref_explor"] = []; meth = []; his = []
    for method in recs[0]["methods"]:
        ms = [r["methods"][method] for r in recs]
        p = torch.stack([m["oof"].float() for m in ms]).mean(0).to(DEV)
        gt = ms[0]["masks"].float().to(DEV); fold = ms[0]["fold"].to(DEV); hi = ms[0]["hi"].to(DEV)
        line = [method]
        for pol in pols:
            tb = best_tb(p, gt, hi, pol); ex = score(p, gt, hi, *tb, pol)
            conf = torch.zeros(len(gt), device=DEV)
            for k in range(R.P.K):
                idx = ms[0]["inner"][k]["idx"].to(DEV)
                ip = torch.stack([m["inner"][k]["probs"].float() for m in ms]).mean(0).to(DEV)
                t = best_tb(ip, gt[idx], hi[idx], pol)
                va = (fold == k).nonzero()[:, 0]
                conf[va] = score(p[va], gt[va], hi[va], *t, pol)
            allv[f"{pol}_explor"] += ex.tolist(); allv[f"{pol}_confirm"] += conf.tolist()
            line.append(f"{pol}: explor {float(ex.mean()):.4f} confirm {float(conf.mean()):.4f} "
                        f"(lo {float(conf[~hi].mean()):.4f} hi {float(conf[hi].mean()):.4f})")
        allv["ref_explor"] += S["per_image"][method]["explor"].tolist(); allv["ref_confirm"] += S["per_image"][method]["confirm"].tolist()
        line.append(f"ref confirm {S['by_method'][method]['confirm']:.4f}")
        meth += [method] * len(gt); his += hi.tolist()
        print(" | ".join(line), flush=True)
    meth = np.array(meth); his = np.array(his)
    for base in ("ref", "n01"):
        for pol in ("rect", "sel"):
            for m in ("explor", "confirm"):
                a = np.array(allv[f"{pol}_{m}"]); b = np.array(allv[f"{base}_{m}"]); dlt = a - b
                lo, hi_ = boot_ci(dlt, meth)
                print(f"{pol} vs {base} [{m}]: {b.mean():.4f} -> {a.mean():.4f} delta {dlt.mean():+.4f} CI95 [{lo:+.4f},{hi_:+.4f}] "
                      f"W/L {(dlt > 1e-9).sum()}/{(dlt < -1e-9).sum()} | lo-regime delta {dlt[~his].mean():+.4f}")
    for m in ("explor", "confirm"):
        a = np.array(allv[f"n01_{m}"]); b = np.array(allv[f"ref_{m}"])
        print(f"n01 vs ref [{m}]: {b.mean():.4f} -> {a.mean():.4f}")
    print("---\nval_miou: 0.000000\nA02 DONE")


if __name__ == "__main__":
    main()
