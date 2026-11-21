"""A04 v2: same crop-decoding verifier as prepare_crops.py but with non-square windows (aspect 1:2, 2:1, 1:4, 4:1;
the task allows patch aspect up to 16:1, 2:1 for TrustMark). Stride = window side / 3. Each window is resized by
the wrapper to the (square) decoder input, i.e. the same anisotropic rescale an embedder applies to a patch.
Channels: one per window shape (max over covering crops of the soft agreement), true and random message.
usage: prepare_crops_rect.py <method> <split> start end [rect|small] -> research_cache/A04/{split}_{method}_{variant}_{start}.pt"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

VARIANT = sys.argv[5] if len(sys.argv) > 5 else "rect"
if VARIANT == "rect":
    SHAPES = [(64, 128), (96, 192), (128, 256), (176, 352), (64, 256), (96, 384)]
    SHAPES = SHAPES + [(w, h) for h, w in SHAPES]
    DIV = 3
else:   # "small": small squares / small 1:2 / thin 1:8 windows (tiny regions, elongated PixelSeal patches)
    SHAPES = [(48, 48), (64, 64), (48, 96), (96, 48), (48, 384), (384, 48), (64, 512), (512, 64)]
    DIV = 2
OUT = Path("/p/project1/training2666/jacquet2/research_cache/A04"); OUT.mkdir(parents=True, exist_ok=True)


def grid(L, s):
    st = max(s // DIV, 8); pos = list(range(0, L - s + 1, st))
    if pos[-1] != L - s:
        pos.append(L - s)
    return pos


BOX = [(y, x, h, w, k) for k, (h, w) in enumerate(SHAPES) for y in grid(512, h) for x in grid(512, w)]
COV = torch.zeros(len(BOX), 32, 32, dtype=torch.bool)
for j, (y, x, h, w, _) in enumerate(BOX):
    COV[j, y // 16:(y + h + 15) // 16, x // 16:(x + w + 15) // 16] = True
SHP = torch.tensor([b[4] for b in BOX])


def crop_logits(wm, x, chunk=192):
    out = []
    for k, (h, w) in enumerate(SHAPES):
        bs = [b for b in BOX if b[4] == k]
        cr = torch.stack([x[:, y:y + h, xx:xx + w] for y, xx, _, _, _ in bs])
        for i in range(0, len(cr), chunk):
            out.append(wm.logits(cr[i:i + chunk]).float().cpu())
    return torch.cat(out)


def to_map(score):
    m = torch.full((len(SHAPES), 32, 32), -1.0)
    for k in range(len(SHAPES)):
        sel = SHP == k
        m[k] = torch.where(COV[sel], score[sel][:, None, None], torch.tensor(-1.0)).max(0).values
    return m


def main():
    method, split = sys.argv[1], sys.argv[2]
    names = P.load_cache(method, split)["names"]
    a, b = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(names))
    out = OUT / f"{split}_{method}_{VARIANT}_{a}.pt"
    print("n boxes", len(BOX), flush=True)
    if out.exists():
        print("exists", out); print("---\nval_miou: 0.000000"); return
    wm = load_watermark(method, P.TASK_ROOT / "models", "cuda")
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    rng = np.random.RandomState(23 + a)
    rec = {k: [] for k in ("soft", "soft_ctrl")}; t0 = time.time()
    for n in names[a:b]:
        it = ds[pos[n]]
        x = torch.from_numpy(np.ascontiguousarray(it["image"])).permute(2, 0, 1).float().div(255).cuda()
        m = torch.from_numpy(np.asarray(it["message"]).astype(np.float32))
        mc = torch.from_numpy(rng.randint(0, 2, m.shape).astype(np.float32))
        with torch.no_grad():
            lg = crop_logits(wm, x)
        for suf, msg in (("", m), ("_ctrl", mc)):
            rec["soft" + suf].append(to_map(torch.tanh(lg * (2 * msg - 1)[None] / 4).mean(1)))
    rec = {k: torch.stack(v) for k, v in rec.items()}; rec["names"] = names[a:b]; rec["shapes"] = SHAPES
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{method} {split} [{a}:{b}] {time.time() - t0:.0f}s -> {out}", flush=True)
    print("---\nval_miou: 0.000000")


if __name__ == "__main__":
    main()
