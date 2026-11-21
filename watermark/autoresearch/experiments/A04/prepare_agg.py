"""A04 v3: square (prepare_crops.py) + non-square (prepare_crops_rect.py 'rect') windows in one pass, with three
aggregations over the windows covering each 16-px cell, per window shape: max (as before), mean and min.
Motivation: max-over-covering only says "some window around this cell carries the message"; in the hi regime (holes in
a full watermark) a hole cell is covered by large watermarked windows, so max cannot see the hole, while mean / min can.
Stores soft_{max,mean,min} [N, 17, 32, 32] (true message) and the same with a random message (_ctrl).
usage: prepare_agg.py <method> <split> start end -> research_cache/A04/{split}_{method}_agg_{start}.pt"""
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

OUT = Path("/p/project1/training2666/jacquet2/research_cache/A04"); OUT.mkdir(parents=True, exist_ok=True)
SQ = [(s, s) for s in (96, 128, 176, 256, 352)]
RECT = [(64, 128), (96, 192), (128, 256), (176, 352), (64, 256), (96, 384)]
SHAPES = [(sh, 4) for sh in SQ] + [(sh, 3) for sh in RECT + [(w, h) for h, w in RECT]]   # (shape, stride divisor)


def grid(L, s, div):
    st = max(s // div, 8); pos = list(range(0, L - s + 1, st))
    if pos[-1] != L - s:
        pos.append(L - s)
    return pos


BOX = [(y, x, h, w, k) for k, ((h, w), dv) in enumerate(SHAPES) for y in grid(512, h, dv) for x in grid(512, w, dv)]
COV = torch.zeros(len(BOX), 32, 32, dtype=torch.bool)
for j, (y, x, h, w, _) in enumerate(BOX):
    COV[j, y // 16:(y + h + 15) // 16, x // 16:(x + w + 15) // 16] = True
SHP = torch.tensor([b[4] for b in BOX])
COVF = COV.float().cuda()


def crop_logits(wm, x, chunk=192):
    out = []
    for k, ((h, w), _) in enumerate(SHAPES):
        cr = torch.stack([x[:, y:y + h, xx:xx + w] for y, xx, _, _, kk in BOX if kk == k])
        for i in range(0, len(cr), chunk):
            out.append(wm.logits(cr[i:i + chunk]).float().cpu())
    return torch.cat(out)


def to_maps(score):
    score = score.cuda(); K = len(SHAPES)
    mx = torch.empty(K, 32, 32, device="cuda"); mn = torch.empty_like(mx); me = torch.empty_like(mx)
    for k in range(K):
        sel = (SHP == k).cuda(); c = COVF[sel]; s = score[sel][:, None, None]
        mx[k] = torch.where(c > 0, s, torch.tensor(-9.0, device="cuda")).max(0).values
        mn[k] = torch.where(c > 0, s, torch.tensor(9.0, device="cuda")).min(0).values
        me[k] = (c * s).sum(0) / c.sum(0).clamp(min=1)
    return mx.cpu(), me.cpu(), mn.cpu()


def main():
    method, split = sys.argv[1], sys.argv[2]
    names = P.load_cache(method, split)["names"]
    a, b = int(sys.argv[3]), int(sys.argv[4])
    out = OUT / f"{split}_{method}_agg_{a}.pt"
    print("n boxes", len(BOX), flush=True)
    if out.exists():
        print("exists", out); print("---\nval_miou: 0.000000"); return
    wm = load_watermark(method, P.TASK_ROOT / "models", "cuda")
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    rng = np.random.RandomState(31 + a)
    rec = {}; t0 = time.time()
    for n in names[a:b]:
        it = ds[pos[n]]
        x = torch.from_numpy(np.ascontiguousarray(it["image"])).permute(2, 0, 1).float().div(255).cuda()
        m = torch.from_numpy(np.asarray(it["message"]).astype(np.float32))
        mc = torch.from_numpy(rng.randint(0, 2, m.shape).astype(np.float32))
        with torch.no_grad():
            lg = crop_logits(wm, x)
        for suf, msg in (("", m), ("_ctrl", mc)):
            for agg, v in zip(("max", "mean", "min"), to_maps(torch.tanh(lg * (2 * msg - 1)[None] / 4).mean(1))):
                rec.setdefault(f"soft_{agg}{suf}", []).append(v)
    rec = {k: torch.stack(v) for k, v in rec.items()}; rec["names"] = names[a:b]; rec["shapes"] = SHAPES
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{method} {split} [{a}:{b}] {time.time() - t0:.0f}s -> {out}", flush=True)
    print("---\nval_miou: 0.000000")


if __name__ == "__main__":
    main()
