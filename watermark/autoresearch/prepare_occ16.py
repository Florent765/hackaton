"""Harness v2b: PixelSeal occlusion on a 16x16 grid (32 px cells) -> <cache>/{split}_pixelseal_occ16.pt
{"names", "del16": [N,1,16,16] float16}. Same definition as prepare_occ.del8 with finer cells (256 forwards per
image, PixelSeal works at 256 px so this is cheap). Idempotent. Run through the worker queue (as train.py)."""
import os
import time

import torch

import prepare as P
from prepare_occ import blur, logits
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

DEV, METHOD, G = "cuda", "pixelseal", 16


@torch.no_grad()
def del_grid(wm, x, sign, g=G):
    agree = lambda lg: (lg * sign).mean(1) / lg.abs().mean(1).clamp(min=1e-6)
    cs = 512 // g; xb = blur(x); a0 = agree(logits(wm, METHOD, x)); out = []
    vs = []
    for r in range(g):
        for c in range(g):
            v = x.clone(); v[..., cs * r:cs * r + cs, cs * c:cs * c + cs] = xb[..., cs * r:cs * r + cs, cs * c:cs * c + cs]
            vs.append(v)
    d = torch.cat([agree(logits(wm, METHOD, torch.cat(vs[j:j + 64]))) for j in range(0, g * g, 64)])
    return (a0 - d).view(1, g, g)


def main():
    wm = load_watermark(METHOD, P.TASK_ROOT / "models", DEV)
    for split in ("train", "test"):
        out = P.CACHE / f"{split}_{METHOD}_occ16.pt"
        if out.exists():
            print("exists", out, flush=True); continue
        ds = LocalisationDataset(P.TASK_ROOT / "release", split)
        names = P.load_cache(METHOD, split)["names"]; pos = {m["image_name"]: i for i, m in enumerate(ds.meta)}
        t0 = time.time(); maps = []
        for n in names:
            it = ds[pos[n]]
            x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(DEV)
            sign = (2 * torch.from_numpy(it["message"]).float() - 1).to(DEV)[None]
            maps.append(del_grid(wm, x, sign).half().cpu())
        tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save({"names": names, "del16": torch.stack(maps)}, tmp)
        os.replace(tmp, out)
        print(f"{split}: {len(names)} maps in {time.time() - t0:.0f}s -> {out}", flush=True)
    print("---\nval_miou: 0.000000\nPREPARE_OCC16 DONE", flush=True)


if __name__ == "__main__":
    main()
