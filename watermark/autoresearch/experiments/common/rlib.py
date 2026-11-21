"""Research library (fixed protocol for E01-E18). Reference = train.py @ 26e5e0d (e29):
per-method SegHead (proj 16 / width 64, dilations 1-2-4-8, grid 64 -> 128), BCE + two-class Dice, 60 epochs,
AdamW 2e-3 / wd 1e-4 / OneCycle, batch 16, log1p(clamp(x, 0)) on extractor maps, agreement + del8 sidecars
(pixelseal, stegastamp), RGB 64 + 2 scalars, 3 folds from the v1 cache, two thresholds by bit accuracy > 0.75.

Differences from train.py, all explicit:
  * sidecars are an explicit list (no silent loading on file presence); a missing declared sidecar raises;
  * one seed per run (`--seed`); seeds are merged later by compare.py (mean of probabilities);
  * for every outer fold, an inner 2-fold CV on the outer training images gives inner predictions used for a
    confirmatory threshold calibration that never sees the outer fold labels;
  * outputs: OOF probs, inner probs, per-fold/seed metadata, timings, peak memory -> <out>/<exp>/seed<s>.pt
Extension points for experiments: `transform` (str or callable), `extra_inputs(c, split) -> dict` (added maps or
replacements), `make_head(maps, n_agree)`. Entry scripts import this file by path; nothing here writes to the
shared caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HARNESS = os.environ.get("AR_HARNESS", "/p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task/work/jacquet2/ar/autoresearch")
sys.path.insert(0, HARNESS)
import prepare as P  # noqa: E402  (fixed harness: folds, caches, official metric)

OUT_ROOT = Path(os.environ.get("AR_RESEARCH_OUT", "/p/project1/training2666/jacquet2/research"))
DEV = "cuda"
REF = dict(epochs=60, batch=16, lr=2e-3, wd=1e-4, dice_w=1.0, proj=16, width=64, head_grid=64,
           transform="current", sidecars={"pixelseal": ["occ"], "stegastamp": ["occ"], "trustmark": []})
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2).tolist()
BACC_SPLIT = 0.75


# ------------------------------------------------------------------ inputs
def tf_current(x):
    return torch.log1p(x.clamp(min=0))


def tf_signed(x):
    return torch.sign(x) * torch.log1p(x.abs())


TRANSFORMS = {"current": tf_current, "signed": tf_signed, "identity": lambda x: x}


def std_map(d):
    return (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)


def load_sidecar(method, split, name, names):
    path = P.CACHE / f"{split}_{method}_{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"declared sidecar missing: {path}")
    o = torch.load(path, map_location="cpu", weights_only=False)
    assert o["names"] == names, f"sidecar {path} not aligned with cache names"
    key = {"occ": "del8", "occ16": "del16"}.get(name, name)
    return o[key].float()


def build_inputs(c, split, cfg, extra_inputs=None):
    """Same construction as train.py.inputs (e29) with explicit sidecars and selectable transform."""
    method = c["method"]
    f = {n: c["feats"][n].float() for n in c["feature_names"]}
    tf = cfg["transform"] if callable(cfg["transform"]) else TRANSFORMS[cfg["transform"]]
    for n in c["feature_names"]:
        f[n] = tf(f[n])
    agree = [c["feats"]["agree"].float()] if "agree" in c["feats"] else []
    for sc in cfg["sidecars"].get(method, []):
        agree.append(F.interpolate(std_map(load_sidecar(method, split, sc, c["names"])), size=32, mode="bilinear",
                                   align_corners=False))
    if agree:
        f["agree"] = torch.cat(agree, 1)
    f["rgb"] = c["feats"]["rgb"].float(); f["scalars"] = c["feats"]["scalars"].float()
    if extra_inputs is not None:
        ex = extra_inputs(c, split)
        if "agree_extra" in ex:                               # channels appended to the agreement stack
            a = ex.pop("agree_extra").float()
            a = F.interpolate(a, size=32, mode="bilinear", align_corners=False) if a.shape[-1] != 32 else a
            f["agree"] = torch.cat([f["agree"], a], 1) if "agree" in f else a
        f.update(ex)
    for k, v in f.items():
        assert torch.isfinite(v).all(), f"non-finite input {k}"
    return {k: v.to(DEV) for k, v in f.items()}


# ------------------------------------------------------------------ head (identical to train.py e29)
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegHead(nn.Module):
    def __init__(self, map_channels, n_agree, n_extra_glob=0, proj=16, width=64, grid=64, use_rgb=True):
        super().__init__()
        self.names, self.grid, self.use_rgb = list(map_channels), grid, use_rgb
        self.proj = nn.ModuleList([ConvBNReLU(c, proj, k=1) for c in map_channels.values()])
        cin = proj * len(map_channels) + n_agree + (3 if use_rgb else 0) + 2
        self.body = nn.Sequential(ConvBNReLU(cin, width), ConvBNReLU(width, width, d=2), ConvBNReLU(width, width, d=4),
                                  ConvBNReLU(width, width, d=8), ConvBNReLU(width, width))
        self.up = nn.Sequential(ConvBNReLU(width + (3 if use_rgb else 0), 32), nn.Conv2d(32, 1, 1))

    def forward(self, f):
        up = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
        G = self.grid
        parts = [up(p(f[n]), G) for p, n in zip(self.proj, self.names)]
        if "agree" in f:
            parts.append(up(f["agree"], G))
        if self.use_rgb:
            parts.append(up(f["rgb"], G))
        parts.append(f["scalars"][:, :, None, None].expand(-1, -1, G, G))
        h = up(self.body(torch.cat(parts, 1)), P.GRID)
        tail = [h] + ([up(f["rgb"], P.GRID)] if self.use_rgb else [])
        return self.up(torch.cat(tail, 1))[:, 0]


def default_head(maps, n_agree, cfg):
    return SegHead(maps, n_agree, proj=cfg["proj"], width=cfg["width"], grid=cfg["head_grid"],
                   use_rgb=cfg.get("use_rgb", True))


def loss_fn(logits, t, dice_w=1.0, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits).flatten(1); t = t.flatten(1); d = 0
    for pc, tc in ((p, t), (1 - p, 1 - t)):
        d = d + (2 * (pc * tc).sum(1) + eps) / (pc.sum(1) + tc.sum(1) + eps)
    return bce + dice_w * (1 - d.mean() / 2)


MAP_EXCLUDE = ("agree", "rgb", "scalars")


def fit(f, y, idx, seed, cfg, make_head=default_head):
    torch.manual_seed(seed)
    maps = {n: f[n].shape[1] for n in f if n not in MAP_EXCLUDE}
    head = make_head(maps, f["agree"].shape[1] if "agree" in f else 0, cfg).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * math.ceil(len(idx) / cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    idx = torch.as_tensor(idx, device=DEV)
    for _ in range(cfg["epochs"]):
        head.train()
        perm = idx[torch.randperm(len(idx), device=DEV)]
        for s in range(0, len(perm), cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            if len(b) < 2:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head({k: v[b] for k, v in f.items()})
            loss = loss_fn(lg.float(), y[b], cfg["dice_w"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head


@torch.no_grad()
def predict(head, f, idx):
    head.eval(); out = []
    idx = torch.as_tensor(idx, device=DEV)
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(head({k: v[b] for k, v in f.items()}).float()))
    return torch.cat(out)


# ------------------------------------------------------------------ metric / thresholds
def miou_batch(pred, gt):
    p = pred.bool().flatten(1); g = gt.bool().flatten(1); s = 0
    for pc, gc in ((~p, ~g), (p, g)):
        i = (pc & gc).sum(1).float(); u = (pc | gc).sum(1).float()
        s = s + torch.where(u == 0, torch.ones_like(u), i / u.clamp(min=1))
    return s / 2


def bin2(p, hi, t):
    return torch.where(hi[:, None, None], p >= t[1], p >= t[0])


def best_two_thr(p, y, hi):
    sc = {(a, b): float(miou_batch(bin2(p, hi, (a, b)), y).mean()) for a in THRESHOLDS for b in THRESHOLDS}
    return max(sc, key=sc.get)


# ------------------------------------------------------------------ one seed of one experiment
def run(exp_id, cfg, seed, extra_inputs=None, make_head=default_head, methods=P.METHODS, notes=""):
    out_dir = OUT_ROOT / exp_id; out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time(); torch.cuda.reset_peak_memory_stats()
    rec = {"exp": exp_id, "seed": seed, "cfg": {k: (v if not callable(v) else v.__name__) for k, v in cfg.items()},
           "notes": notes, "methods": {}, "harness_cache": str(P.CACHE)}
    for method in methods:
        t0 = time.time()
        c = P.load_cache(method); f = build_inputs(c, "train", cfg, extra_inputs)
        t_inputs = time.time() - t0
        y = c["masks"].to(DEV).float(); fold = c["fold"]
        hi = (c["feats"]["scalars"][:, 1] > BACC_SPLIT).to(DEV)
        oof = torch.zeros_like(y); inner = {}; t_fit = 0.0; n_fit = 0
        for k in range(P.K):
            tr = torch.nonzero(fold != k)[:, 0]; va = torch.nonzero(fold == k)[:, 0]
            t1 = time.time()
            oof[va] = predict(fit(f, y, tr.tolist(), seed, cfg, make_head), f, va.tolist())
            # inner 2-fold CV on the outer training images (labels of fold k never used)
            ip = torch.zeros(len(tr), *y.shape[1:], device=DEV)
            sub = fold[tr]
            for j in sub.unique().tolist():
                a = tr[sub != j]; b = tr[sub == j]
                ip[(sub == j).nonzero()[:, 0]] = predict(fit(f, y, a.tolist(), seed, cfg, make_head), f, b.tolist())
            inner[k] = {"idx": tr.cpu(), "probs": ip.half().cpu()}
            t_fit += time.time() - t1; n_fit += 3
        rec["methods"][method] = {"names": c["names"], "oof": oof.half().cpu(), "inner": inner,
                                  "masks": c["masks"].clone(),
                                  "fold": fold.clone(), "hi": hi.cpu(), "full_watermark": c["full_watermark"].clone(),
                                  "t_inputs": t_inputs, "t_fit": t_fit, "n_fit": n_fit,
                                  "in_shapes": {k: tuple(v.shape[1:]) for k, v in f.items()}}
        print(f"[{exp_id} s{seed}] {method}: inputs {t_inputs:.1f}s, {n_fit} fits {t_fit:.0f}s, "
              f"in {rec['methods'][method]['in_shapes']}", flush=True)
        del f, y; torch.cuda.empty_cache()
    rec["t_total"] = time.time() - t_start
    rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    path = out_dir / f"seed{seed}.pt"; tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(rec, tmp); os.replace(tmp, path)
    s = summarize([rec])
    print(json.dumps(s["overall"]), flush=True)
    print(f"---\nval_miou: {s['overall']['explor']:.6f}\nconfirm_miou: {s['overall']['confirm']:.6f}\n"
          f"fixed05_miou: {s['overall']['fixed05']:.6f}\ntotal_seconds: {rec['t_total']:.1f}\n"
          f"peak_vram_mb: {rec['peak_mem_mb']:.1f}", flush=True)
    return rec


# ------------------------------------------------------------------ evaluation of one or several seeds
def summarize(recs):
    """recs: list of seed records of the same experiment -> per-image scores for the three threshold policies."""
    out = {"per_image": {}, "overall": {}, "by_method": {}, "thr": {}}
    allv = {"fixed05": [], "explor": [], "confirm": []}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for method in recs[0]["methods"]:
        ms = [r["methods"][method] for r in recs]
        p = torch.stack([m["oof"].float() for m in ms]).mean(0).to(dev)
        gt = (ms[0]["masks"] if "masks" in ms[0] else P.load_cache(method)["masks"]).float().to(dev)
        hi = ms[0]["hi"].to(dev); fold = ms[0]["fold"].to(dev)
        res = {"fixed05": miou_batch(p >= 0.5, gt)}
        t_ex = best_two_thr(p, gt, hi); res["explor"] = miou_batch(bin2(p, hi, t_ex), gt)
        conf = torch.zeros(len(gt), device=dev); t_conf = {}
        for k in range(P.K):
            idx = ms[0]["inner"][k]["idx"].to(dev)
            ip = torch.stack([m["inner"][k]["probs"].float() for m in ms]).mean(0).to(dev)
            t = best_two_thr(ip, gt[idx], hi[idx]); t_conf[k] = t
            va = (fold == k).nonzero()[:, 0]
            conf[va] = miou_batch(bin2(p[va], hi[va], t), gt[va])
        res["confirm"] = conf
        res = {k: v.cpu() for k, v in res.items()}
        for key in allv:
            allv[key] += res[key].tolist()
        cov = gt.flatten(1).mean(1).cpu()
        out["per_image"][method] = {"names": ms[0]["names"], "cov": cov, **res}
        out["by_method"][method] = {k: float(v.mean()) for k, v in res.items()}
        out["thr"][method] = {"explor": t_ex, "confirm": t_conf}
    out["overall"] = {k: float(np.mean(v)) for k, v in allv.items()}
    return out


def main_cli(exp_id, cfg, extra_inputs=None, make_head=default_head, notes=""):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(exp_id, cfg, a.seed, extra_inputs, make_head, notes=notes)
