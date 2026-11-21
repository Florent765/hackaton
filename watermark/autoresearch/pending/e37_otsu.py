# PENDING EXPERIMENT e37_otsu: per-image threshold = alpha*Otsu(p_i) + (1-alpha)*mode threshold, alpha chosen on OOF
"""autoresearch: the ONLY file the agent edits. Trains one head per method on cached frozen features, K-fold,
reports out-of-fold official mIoU. `--final` additionally fits on all train images and writes a test submission.
Everything here is fair game (inputs used, head, loss, optimiser, epochs, post-processing, thresholds)."""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare as P

# ---------------------------------------------------------------- hyperparameters
EPOCHS = 60
BATCH = 16
LR = 2e-3
WD = 1e-4
DICE_W = 1.0
PROJ, WIDTH = 16, 64
HEAD_GRID = 64
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2).tolist()
SEED = 0
DEV = "cuda"


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegHead(nn.Module):
    def __init__(self, map_channels, n_agree):
        super().__init__()
        self.names = list(map_channels)
        self.proj = nn.ModuleList([ConvBNReLU(c, PROJ, k=1) for c in map_channels.values()])
        cin = PROJ * len(map_channels) + n_agree + 3 + 2
        self.body = nn.Sequential(ConvBNReLU(cin, WIDTH), ConvBNReLU(WIDTH, WIDTH, d=2), ConvBNReLU(WIDTH, WIDTH, d=4),
                                  ConvBNReLU(WIDTH, WIDTH, d=8), ConvBNReLU(WIDTH, WIDTH))
        self.up = nn.Sequential(ConvBNReLU(WIDTH + 3, 32), nn.Conv2d(32, 1, 1))

    def forward(self, f):
        up = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
        parts = [up(p(f[n]), HEAD_GRID) for p, n in zip(self.proj, self.names)]
        if "agree" in f:
            parts.append(up(f["agree"], HEAD_GRID))
        parts += [f["rgb"], f["scalars"][:, :, None, None].expand(-1, -1, HEAD_GRID, HEAD_GRID)]
        h = up(self.body(torch.cat(parts, 1)), P.GRID)
        return self.up(torch.cat([h, up(f["rgb"], P.GRID)], 1))[:, 0]


def loss_fn(logits, t, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits).flatten(1); t = t.flatten(1); d = 0
    for pc, tc in ((p, t), (1 - p, 1 - t)):
        d = d + (2 * (pc * tc).sum(1) + eps) / (pc.sum(1) + tc.sum(1) + eps)
    return bce + DICE_W * (1 - d.mean() / 2)


def inputs(c, split="train"):
    """Select / transform cached inputs (on GPU, float32)."""
    f = {n: c["feats"][n] for n in c["feature_names"]}
    occ_path = P.CACHE / f"{split}_{c['method']}_occ.pt"
    if occ_path.exists():                                   # harness v2: occlusion maps (pixelseal, stegastamp)
        o = torch.load(occ_path, map_location="cpu", weights_only=False)
        assert o["names"] == c["names"]
        d = o["del8"].float()
        d = (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)
        d = F.interpolate(d, size=32, mode="bilinear", align_corners=False)
        f["agree"] = torch.cat([c["feats"]["agree"].float(), d], 1) if "agree" in c["feats"] else d
    for k in ("agree", "rgb", "scalars"):
        if k in c["feats"] and k not in f:
            f[k] = c["feats"][k]
    out = {k: v.to(DEV).float() for k, v in f.items()}
    for n in c["feature_names"]:
        out[n] = torch.log1p(out[n].clamp(min=0))
    return out


def fit(f, y, idx, seed=SEED):
    torch.manual_seed(seed)
    maps = {n: f[n].shape[1] for n in f if n not in ("agree", "rgb", "scalars")}
    head = SegHead(maps, f["agree"].shape[1] if "agree" in f else 0).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(idx) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    idx = torch.as_tensor(idx, device=DEV)
    for _ in range(EPOCHS):
        head.train()
        perm = idx[torch.randperm(len(idx), device=DEV)]
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 2:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head({k: v[b] for k, v in f.items()})
            loss = loss_fn(lg.float(), y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head


@torch.no_grad()
def probs(head, f, idx):
    head.eval(); out = []
    idx = torch.as_tensor(idx, device=DEV)
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(head({k: v[b] for k, v in f.items()}).float()))
    return torch.cat(out)


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def otsu(p, bins=64):
    """Per-image Otsu threshold of the probability map. p: [N,H,W] in [0,1] -> [N]."""
    out = []
    centers = (torch.arange(bins, device=p.device).float() + 0.5) / bins
    for x in p:
        h = torch.histc(x.float(), bins, 0, 1); h = h / h.sum().clamp(min=1)
        w0 = h.cumsum(0); w1 = 1 - w0
        m0 = (h * centers).cumsum(0); mt = m0[-1]
        var = (mt * w0 - m0) ** 2 / (w0 * w1).clamp(min=1e-9)
        out.append(centers[var.argmax()])
    return torch.stack(out)


def miou_batch(pred, gt):
    p = pred.bool().flatten(1); g = gt.bool().flatten(1); s = 0
    for pc, gc in ((~p, ~g), (p, g)):
        i = (pc & gc).sum(1).float(); u = (pc | gc).sum(1).float()
        s = s + torch.where(u == 0, torch.ones_like(u), i / u.clamp(min=1))
    return s / 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", default="")
    ap.add_argument("--save-oof", default="", help="save out-of-fold probabilities for error analysis")
    a = ap.parse_args()
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    oof, thr = {}, {}
    for method in P.METHODS:
        c = P.load_cache(method); f = inputs(c)
        y = c["masks"].to(DEV).float(); fold = c["fold"]
        pr = torch.zeros_like(y)
        for k in range(P.K):
            tr = torch.nonzero(fold != k)[:, 0].tolist(); va = torch.nonzero(fold == k)[:, 0].tolist()
            pr[va] = sum(probs(fit(f, y, tr, seed=s), f, va) for s in range(3)) / 3
        hi = (c["feats"]["scalars"][:, 1] > 0.75).to(DEV)
        ot = otsu(pr)
        def bin2(p, t, hi=hi, ot=ot):
            base = torch.where(hi, torch.full_like(ot, t[1]), torch.full_like(ot, t[0]))
            return p >= (t[2] * ot + (1 - t[2]) * base)[:, None, None]
        sc = {(t0, t1, al): float(miou_batch(bin2(pr, (t0, t1, al)), y).mean())
              for t0 in THRESHOLDS for t1 in THRESHOLDS for al in ALPHAS}
        thr[method] = max(sc, key=sc.get)
        oof[method] = bin2(pr, thr[method]).to(torch.uint8).cpu()
        if a.save_oof:
            oof.setdefault("_probs", {})[method] = {"probs": pr.half().cpu(), "names": c["names"], "thr": thr[method],
                                                    "scalars": c["feats"]["scalars"]}
        print(f"{method}: thr {thr[method]} oof mIoU {sc[thr[method]]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if a.final:
            ct = P.load_cache(method, "test"); ft = inputs(ct, "test")
            pt = sum(probs(fit(f, y, list(range(len(y))), seed=s), ft, list(range(len(ct["names"])))) for s in range(3)) / 3
            hit = (ct["feats"]["scalars"][:, 1] > 0.75).to(DEV)
            mt = bin2(pt, thr[method], hi=hit, ot=otsu(pt))
            oof.setdefault("_test", []).append((ct["names"], mt.cpu()))
            del ft
        del f, y
        torch.cuda.empty_cache()
    res = P.evaluate_oof({m: oof[m] for m in P.METHODS})
    if a.save_oof:
        torch.save(oof["_probs"], a.save_oof)
    if a.final:
        names = sum([n for n, _ in oof["_test"]], [])
        masks = torch.cat([m for _, m in oof["_test"]])
        size = P.write_submission(names, masks, a.final)
        print(f"submission {a.final} {size:.2f} MB", flush=True)
    print("---")
    print(f"val_miou:         {res['val_miou']:.6f}")
    for m in P.METHODS:
        print(f"miou_{m}: {res[m]:.6f}  thr {thr[m]}")
    print(f"total_seconds:    {time.time() - t0:.1f}")
    print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 2**20:.1f}")


if __name__ == "__main__":
    main()
