"""Harness v2 extension (human decision, MEGA_PROMPT step B): occlusion maps with the known message.
Writes <cache>/{split}_{method}_occ.pt = {"names": [...], "del8": [N,1,8,8] float16} for pixelseal and stegastamp
(probe_v2: per-image AUROC 0.61 vs 0.59 agreement for PixelSeal; first spatial signal for StegaStamp; useless for
TrustMark, so skipped). del8 = A(x) - A(x with one 64 px cell Gaussian-blurred, sigma 4), A = soft agreement
mean_k (2m_k-1) l_k / mean_k |l_k|.  The v1 caches are not modified.
Run through the worker queue (as train.py) or a GPU job; idempotent (skips existing files)."""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import prepare as P
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

DEV = "cuda"
METHODS = ("pixelseal", "stegastamp")


def blur(x, s=4.0):
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def logits(wm, method, x):
    return wm.logits(x.cpu(), bs=64).to(DEV) if method == "stegastamp" else wm.logits(x)


@torch.no_grad()
def del8(wm, method, x, sign):
    agree = lambda lg: (lg * sign).mean(1) / lg.abs().mean(1).clamp(min=1e-6)
    xb = blur(x); a0 = agree(logits(wm, method, x))
    vs = []
    for r in range(8):
        for c in range(8):
            v = x.clone(); v[..., 64 * r:64 * r + 64, 64 * c:64 * c + 64] = xb[..., 64 * r:64 * r + 64, 64 * c:64 * c + 64]
            vs.append(v)
    d = torch.cat([agree(logits(wm, method, torch.cat(vs[j:j + 32]))) for j in range(0, 64, 32)])
    return (a0 - d).view(1, 8, 8)


def main():
    for split in ("train", "test"):
        ds = LocalisationDataset(P.TASK_ROOT / "release", split)
        for method in METHODS:
            out = P.CACHE / f"{split}_{method}_occ.pt"
            if out.exists():
                print("exists", out, flush=True); continue
            names = P.load_cache(method, split)["names"]                     # same order as the v1 cache
            pos = {m["image_name"]: i for i, m in enumerate(ds.meta)}
            wm = load_watermark(method, P.TASK_ROOT / "models", DEV)
            t0 = time.time(); maps = []
            for n in names:
                it = ds[pos[n]]
                x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(DEV)
                sign = (2 * torch.from_numpy(it["message"]).float() - 1).to(DEV)[None]
                maps.append(del8(wm, method, x, sign).half().cpu())
            rec = {"names": names, "del8": torch.stack(maps)}
            tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
            print(f"{split} {method}: {len(names)} maps in {time.time() - t0:.0f}s -> {out}", flush=True)
            del wm; torch.cuda.empty_cache()
    print("---\nval_miou: 0.000000\nPREPARE_OCC DONE", flush=True)


if __name__ == "__main__":
    main()
