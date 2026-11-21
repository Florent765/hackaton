"""MEGA_PROMPT step B probe: are occlusion maps / gradient saliency more discriminative than the agreement map?
Run through the worker queue (as train.py). For N train images per method it measures, per image, the AUROC
between a candidate map and the ground-truth mask (map upsampled to 128, cells labelled by mask128), plus time.

Candidate maps (known message m, soft agreement A(x) = mean_k (2m_k-1) l_k(x) / mean_k |l_k(x)|):
  agree     current input: per-position agreement from the cache (32x32, PixelSeal/TrustMark only)
  del8      A(x) - A(x with cell blurred), 8x8 grid of 64 px cells  (watermarked cell -> agreement drops)
  keep7     A(only a 128 px window kept, rest blurred), 7x7 windows, stride 64
  grad      |d A / d x| summed over RGB, avg-pooled to 32x32 (torch extractors only)
Blur = Gaussian sigma 4 px (removes the high-frequency watermark, keeps content)."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prepare as P
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

N = int(os.environ.get("PROBE_N", "24"))
DEV = "cuda"


def blur(x, s=4.0):
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = s.argsort().argsort() + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def map_auc(m, mask128):
    up = F.interpolate(torch.as_tensor(m, dtype=torch.float32)[None, None], size=128, mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    return auroc(up, mask128)


class Scorer:
    def __init__(self, method):
        self.method = method
        self.wm = load_watermark(method, P.TASK_ROOT / "models", DEV)

    def logits(self, x):                       # x [B,3,512,512] on DEV in [0,1]
        if self.method == "stegastamp":
            return self.wm.logits(x.cpu(), bs=64).to(DEV)
        return self.wm.logits(x)

    def logits_grad(self, x):                  # differentiable path (torch extractors)
        if self.method == "pixelseal":
            return self.wm.det(self.wm._resize(x))[:, 1:].float()
        return self.wm.net(self.wm._prep(x)).float()


def agreement(lg, sign):
    return (lg * sign).mean(1) / lg.abs().mean(1).clamp(min=1e-6)


def main():
    ds = LocalisationDataset(P.TASK_ROOT / "release", "train")
    res = {}
    for method in P.METHODS:
        cache = P.load_cache(method)
        name_to_ci = {n: i for i, n in enumerate(cache["names"])}
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == method]
        rng = np.random.RandomState(0); idx = list(rng.choice(idx, size=min(N, len(idx)), replace=False))
        sc = Scorer(method); torch.cuda.synchronize()
        aucs = {k: [] for k in ("agree", "del8", "keep7", "grad")}; tt = {k: 0.0 for k in aucs}
        for i in idx:
            it = ds[i]
            x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(DEV)
            sign = (2 * torch.from_numpy(it["message"]).float() - 1).to(DEV)[None]
            gt = it["mask128"]; xb = blur(x)
            if "agree" in cache["feats"]:
                aucs["agree"].append(map_auc(cache["feats"]["agree"][name_to_ci[it["image_name"]], 0].float(), gt))
            with torch.no_grad():
                t0 = time.time(); a0 = agreement(sc.logits(x), sign)
                vs = []
                for r in range(8):
                    for c in range(8):
                        v = x.clone(); v[..., 64 * r:64 * r + 64, 64 * c:64 * c + 64] = xb[..., 64 * r:64 * r + 64, 64 * c:64 * c + 64]
                        vs.append(v)
                d = torch.cat([agreement(sc.logits(torch.cat(vs[j:j + 16])), sign) for j in range(0, 64, 16)])
                torch.cuda.synchronize(); tt["del8"] += time.time() - t0
                aucs["del8"].append(map_auc((a0 - d).view(8, 8).cpu(), gt))
                t0 = time.time(); vs = []
                for r in range(7):
                    for c in range(7):
                        v = xb.clone(); v[..., 64 * r:64 * r + 128, 64 * c:64 * c + 128] = x[..., 64 * r:64 * r + 128, 64 * c:64 * c + 128]
                        vs.append(v)
                k = torch.cat([agreement(sc.logits(torch.cat(vs[j:j + 16])), sign) for j in range(0, 49, 16)])
                torch.cuda.synchronize(); tt["keep7"] += time.time() - t0
                aucs["keep7"].append(map_auc(k.view(7, 7).cpu(), gt))
            if method != "stegastamp":
                t0 = time.time()
                xg = x.clone().requires_grad_(True)
                agreement(sc.logits_grad(xg), sign).sum().backward()
                g = F.avg_pool2d(xg.grad.abs().sum(1, keepdim=True), 16)[0, 0]
                torch.cuda.synchronize(); tt["grad"] += time.time() - t0
                aucs["grad"].append(map_auc(g.cpu(), gt))
        line = {k: (float(np.nanmean(v)) if len(v) else None, round(tt[k] / len(idx), 3)) for k, v in aucs.items()}
        res[method] = line
        print(method, "mean per-image AUROC (s/img):", line, flush=True)
        del sc; torch.cuda.empty_cache()
    print("PROBE_V2 DONE", res, flush=True)


if __name__ == "__main__":
    main()
