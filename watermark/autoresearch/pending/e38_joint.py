# PENDING EXPERIMENT e38_joint: one head shared by the 3 methods (method-specific projections, method one-hot)
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


def miou_batch(pred, gt):
    p = pred.bool().flatten(1); g = gt.bool().flatten(1); s = 0
    for pc, gc in ((~p, ~g), (p, g)):
        i = (pc & gc).sum(1).float(); u = (pc | gc).sum(1).float()
        s = s + torch.where(u == 0, torch.ones_like(u), i / u.clamp(min=1))
    return s / 2


# ---------------- e38: one head shared by the 3 methods (method-specific input projections)
N_AGREE = 3


class JointHead(nn.Module):
    def __init__(self, maps_per_method):
        super().__init__()
        self.names = {m: list(mc) for m, mc in maps_per_method.items()}
        self.proj = nn.ModuleDict({m: nn.ModuleList([ConvBNReLU(c, PROJ, k=1) for c in mc.values()])
                                   for m, mc in maps_per_method.items()})
        cin = PROJ * 2 + N_AGREE + 3 + 2 + len(P.METHODS)
        self.body = nn.Sequential(ConvBNReLU(cin, WIDTH), ConvBNReLU(WIDTH, WIDTH, d=2), ConvBNReLU(WIDTH, WIDTH, d=4),
                                  ConvBNReLU(WIDTH, WIDTH, d=8), ConvBNReLU(WIDTH, WIDTH))
        self.up = nn.Sequential(ConvBNReLU(WIDTH + 3, 32), nn.Conv2d(32, 1, 1))

    def forward(self, f, method):
        up = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
        parts = [up(p(f[n]), HEAD_GRID) for p, n in zip(self.proj[method], self.names[method])]
        B = parts[0].shape[0]
        ag = up(f["agree"], HEAD_GRID) if "agree" in f else parts[0].new_zeros(B, 0, HEAD_GRID, HEAD_GRID)
        ag = torch.cat([ag, ag.new_zeros(B, N_AGREE - ag.shape[1], HEAD_GRID, HEAD_GRID)], 1)
        onehot = parts[0].new_zeros(B, len(P.METHODS), HEAD_GRID, HEAD_GRID)
        onehot[:, P.METHODS.index(method)] = 1
        parts += [ag, f["rgb"], f["scalars"][:, :, None, None].expand(-1, -1, HEAD_GRID, HEAD_GRID), onehot]
        h = up(self.body(torch.cat(parts, 1)), P.GRID)
        return self.up(torch.cat([h, up(f["rgb"], P.GRID)], 1))[:, 0]


def fit_joint(fs, ys, idxs, seed=SEED):
    torch.manual_seed(seed)
    maps = {m: {n: fs[m][n].shape[1] for n in fs[m] if n not in ("agree", "rgb", "scalars")} for m in fs}
    head = JointHead(maps).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    per_epoch = sum(math.ceil(len(idxs[m]) / BATCH) for m in fs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * per_epoch, pct_start=0.1)
    idxs = {m: torch.as_tensor(v, device=DEV) for m, v in idxs.items()}
    for _ in range(EPOCHS):
        head.train()
        batches = []
        for m in fs:
            perm = idxs[m][torch.randperm(len(idxs[m]), device=DEV)]
            batches += [(m, perm[s:s + BATCH]) for s in range(0, len(perm), BATCH)]
        for j in torch.randperm(len(batches)).tolist():
            m, b = batches[j]
            if len(b) < 2:
                sched.step(); continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head({k: v[b] for k, v in fs[m].items()}, m)
            loss = loss_fn(lg.float(), ys[m][b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head


@torch.no_grad()
def probs_joint(head, f, idx, method):
    head.eval(); out = []
    idx = torch.as_tensor(idx, device=DEV)
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(head({k: v[b] for k, v in f.items()}, method).float()))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", default="")
    ap.add_argument("--save-oof", default="")
    a = ap.parse_args()
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    cs = {m: P.load_cache(m) for m in P.METHODS}
    fs = {m: inputs(cs[m]) for m in P.METHODS}
    ys = {m: cs[m]["masks"].to(DEV).float() for m in P.METHODS}
    prs = {m: torch.zeros_like(ys[m]) for m in P.METHODS}
    for k in range(P.K):
        tr = {m: torch.nonzero(cs[m]["fold"] != k)[:, 0].tolist() for m in P.METHODS}
        va = {m: torch.nonzero(cs[m]["fold"] == k)[:, 0].tolist() for m in P.METHODS}
        heads = [fit_joint(fs, ys, tr, seed=s) for s in range(3)]
        for m in P.METHODS:
            prs[m][va[m]] = sum(probs_joint(h, fs[m], va[m], m) for h in heads) / 3
        print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    oof, thr, test = {}, {}, []
    if a.final:
        heads = [fit_joint(fs, ys, {m: list(range(len(ys[m]))) for m in P.METHODS}, seed=s) for s in range(3)]
    for m in P.METHODS:
        pr, y = prs[m], ys[m]
        hi = (cs[m]["feats"]["scalars"][:, 1] > 0.75).to(DEV)
        bin2 = lambda p, t, hi=hi: torch.where(hi[:, None, None], p >= t[1], p >= t[0])
        sc = {(t0_, t1_): float(miou_batch(bin2(pr, (t0_, t1_)), y).mean()) for t0_ in THRESHOLDS for t1_ in THRESHOLDS}
        thr[m] = max(sc, key=sc.get)
        oof[m] = bin2(pr, thr[m]).to(torch.uint8).cpu()
        print(f"{m}: thr {thr[m]} oof mIoU {sc[thr[m]]:.4f}", flush=True)
        if a.final:
            ct = P.load_cache(m, "test"); ft = inputs(ct, "test"); n = list(range(len(ct["names"])))
            pt = sum(probs_joint(h, ft, n, m) for h in heads) / 3
            hit = (ct["feats"]["scalars"][:, 1] > 0.75).to(DEV)
            test.append((ct["names"], torch.where(hit[:, None, None], pt >= thr[m][1], pt >= thr[m][0]).cpu()))
            del ft
    res = P.evaluate_oof(oof)
    if a.final:
        size = P.write_submission(sum([n for n, _ in test], []), torch.cat([t for _, t in test]), a.final)
        print(f"submission {a.final} {size:.2f} MB", flush=True)
    print("---")
    print(f"val_miou:         {res['val_miou']:.6f}")
    for m in P.METHODS:
        print(f"miou_{m}: {res[m]:.6f}  thr {thr[m]}")
    print(f"total_seconds:    {time.time() - t0:.1f}")
    print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 2**20:.1f}")


if __name__ == "__main__":
    main()
