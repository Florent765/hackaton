"""A04: multi-scale crop-decoding maps (decoder used as a sliding verifier), cached as head inputs.
For square windows s in SIZES (stride s/4, 512 frame) every crop is decoded (wrapper logits(): resize to the
decoder resolution) and scored with the known message: soft agreement a = mean_k tanh(s_k * l_k / 4) and bit accuracy.
Per 32x32 cell (16 px) and per scale: max over the crops covering the cell -> channels [N, len(SIZES), 32, 32],
for the true message ('true') and a fixed random message ('ctrl').
usage: prepare_crops.py <method> <split> [start end]  -> research_cache/A04/{split}_{method}_crops[_{start}].pt"""
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

SIZES = (96, 128, 176, 256, 352)
OUT = Path("/p/project1/training2666/jacquet2/research_cache/A04"); OUT.mkdir(parents=True, exist_ok=True)


def boxes():
    out = []
    for s in SIZES:
        st = s // 4; pos = list(range(0, 512 - s + 1, st))
        if pos[-1] != 512 - s:
            pos.append(512 - s)
        out += [(y, x, s) for y in pos for x in pos]
    return out


BOX = boxes()
# cover[b] : boolean [32,32] of cells (16 px) fully or partly inside box b ; scale index per box
COV = torch.zeros(len(BOX), 32, 32, dtype=torch.bool)
for j, (y, x, s) in enumerate(BOX):
    COV[j, y // 16:(y + s + 15) // 16, x // 16:(x + s + 15) // 16] = True
SCALE = torch.tensor([SIZES.index(s) for _, _, s in BOX])


def crop_logits(wm, x, chunk=192):
    crops = []
    out = []
    for s in SIZES:
        bs = [b for b in BOX if b[2] == s]
        cr = torch.stack([x[:, y:y + s, xx:xx + s] for y, xx, _ in bs])
        for i in range(0, len(cr), chunk):
            out.append(wm.logits(cr[i:i + chunk]).float().cpu())
    return torch.cat(out)


def to_map(score):
    """score [n_boxes] -> [len(SIZES), 32, 32] max over covering crops of each scale"""
    m = torch.full((len(SIZES), 32, 32), -1.0)
    for k in range(len(SIZES)):
        sel = SCALE == k
        v = torch.where(COV[sel], score[sel][:, None, None], torch.tensor(-1.0))
        m[k] = v.max(0).values
    return m


def main():
    method, split = sys.argv[1], sys.argv[2]
    names = P.load_cache(method, split)["names"]
    a, b = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(names))
    out = OUT / (f"{split}_{method}_crops.pt" if (a, b) == (0, len(names)) else f"{split}_{method}_crops_{a}.pt")
    if out.exists():
        print("exists", out); print("---\nval_miou: 0.000000"); return
    wm = load_watermark(method, P.TASK_ROOT / "models", "cuda")
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    rng = np.random.RandomState(11 + a)
    rec = {k: [] for k in ("soft", "ba", "soft_ctrl", "ba_ctrl")}; t0 = time.time()
    for n in names[a:b]:
        it = ds[pos[n]]
        x = torch.from_numpy(it["image"]).permute(2, 0, 1).float().div(255).cuda()
        m = torch.from_numpy(np.asarray(it["message"]).astype(np.float32))
        mc = torch.from_numpy(rng.randint(0, 2, m.shape).astype(np.float32))
        with torch.no_grad():
            lg = crop_logits(wm, x)
        for suf, msg in (("", m), ("_ctrl", mc)):
            sg = 2 * msg - 1
            rec["soft" + suf].append(to_map(torch.tanh(lg * sg[None] / 4).mean(1)))
            rec["ba" + suf].append(to_map(((lg > 0).float() == msg[None]).float().mean(1)))
    rec = {k: torch.stack(v) for k, v in rec.items()}; rec["names"] = names[a:b]; rec["sizes"] = SIZES
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{method} {split} [{a}:{b}] {time.time() - t0:.0f}s -> {out}", flush=True)
    print("---\nval_miou: 0.000000")


if __name__ == "__main__":
    main()
