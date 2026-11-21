"""A10: per-window verification statistics + test-time rectangle refinement by decoding (crop verifier, v4).
Window set = the 17 A04 shapes (5 squares stride s/4, 12 non-square stride side/3; 1757 windows on the 512 frame).
For every window and for the true message m (and a fixed random message, control) we store the sufficient statistics
of an exact null-calibrated score:
    S1 = sum_k s_k tanh(l_k/4)   (s = 2m-1),     S2 = sum_k tanh(l_k/4)^2,     z = S1 / sqrt(S2)
Under a uniformly random message (bits independent of the logits) E[z] = 0 and Var[z] = 1 whatever the logits, so z is
comparable across window shapes, decoders and images (the A04 'soft' mean is not: it scales with logit magnitude).
Refinement (new): from the top-4 grid windows (greedy NMS IoU < 0.5) we run a coordinate ascent on z over boxes
(t, l, b, r): 12 moves (each edge +-d, shift +-d vertically / horizontally), d = 48, 24, 12, <= 3 accepted rounds per d.
Refinement crops are resized to 256x256 (bilinear, antialias) before the wrapper, starts are re-scored on that path.
Output research_cache/A10/{split}_{method}_win_{start}.pt with, per message suffix ('' / '_ctrl'):
    s1, s2 [N, 1757] fp16 ; rbox [N, 4, 4] (t, l, b, r px, refined) ; rz [N, 4] refined z ; rz0 [N, 4] start z.
usage: prepare_win.py <method> <split> start end"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/A10"); OUT.mkdir(parents=True, exist_ok=True)
SQ = [(s, s) for s in (96, 128, 176, 256, 352)]
RECT = [(64, 128), (96, 192), (128, 256), (176, 352), (64, 256), (96, 384)]
SHAPES = [(sh, 4) for sh in SQ] + [(sh, 3) for sh in RECT + [(w, h) for h, w in RECT]]   # same as A04/prepare_agg.py
NSTART, DS, ROUNDS, RS = 4, (48, 24, 12), 3, 256
MOVES = [(dt, dl, db, dr) for dt, dl, db, dr in
         [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1),
          (1, 0, 1, 0), (-1, 0, -1, 0), (0, 1, 0, 1), (0, -1, 0, -1)]]


def grid(L, s, div):
    st = max(s // div, 8); pos = list(range(0, L - s + 1, st))
    if pos[-1] != L - s:
        pos.append(L - s)
    return pos


BOX = [(y, x, h, w, k) for k, ((h, w), dv) in enumerate(SHAPES) for y in grid(512, h, dv) for x in grid(512, w, dv)]
TLBR = torch.tensor([(y, x, y + h, x + w) for y, x, h, w, _ in BOX], dtype=torch.float32)


def crop_logits(wm, x, chunk=192):
    out = []
    for k, ((h, w), _) in enumerate(SHAPES):
        cr = torch.stack([x[:, y:y + h, xx:xx + w] for y, xx, _, _, kk in BOX if kk == k])
        for i in range(0, len(cr), chunk):
            out.append(wm.logits(cr[i:i + chunk]).float().cpu())
    return torch.cat(out)


def stats(lg, msg):
    t = torch.tanh(lg / 4)
    return (t * (2 * msg - 1)).sum(-1), (t * t).sum(-1)


def zscore(lg, msg):
    s1, s2 = stats(lg, msg)
    return s1 / s2.clamp(min=1e-6).sqrt()


def box_logits(wm, x, boxes):
    cr = torch.cat([F.interpolate(x[None, :, t:b, l:r], size=(RS, RS), mode="bilinear", antialias=True,
                                  align_corners=False) for t, l, b, r in boxes])
    return wm.logits(cr).float().cpu()


def iou(a, b):
    ih = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    iw = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = ih * iw; ar = lambda q: (q[..., 2] - q[..., 0]) * (q[..., 3] - q[..., 1])
    return inter / (ar(a) + ar(b) - inter)


def starts(z):
    order = torch.argsort(z, descending=True); keep = []
    for j in order.tolist():
        if all(float(iou(TLBR[j], TLBR[i:i + 1])[0]) < 0.5 for i in keep):
            keep.append(j)
        if len(keep) == NSTART:
            break
    return keep


def valid(b):
    t, l, bb, r = b; h, w = bb - t, r - l
    return t >= 0 and l >= 0 and bb <= 512 and r <= 512 and h >= 32 and w >= 32 and max(h / w, w / h) <= 16


def refine(wm, x, msg, init):
    cur = [list(map(int, TLBR[j].tolist())) for j in init]
    zc = zscore(box_logits(wm, x, cur), msg); z0 = zc.clone()
    for d in DS:
        for _ in range(ROUNDS):
            cand = []
            for b in cur:
                for mv in MOVES:
                    nb = [b[i] + d * mv[i] for i in range(4)]
                    cand.append(nb if valid(nb) else b)
            zz = zscore(box_logits(wm, x, cand), msg).view(len(cur), len(MOVES))
            best = zz.argmax(1); moved = False
            for i in range(len(cur)):
                if zz[i, best[i]] > zc[i] + 1e-4:
                    cur[i] = cand[i * len(MOVES) + int(best[i])]; zc[i] = zz[i, best[i]]; moved = True
            if not moved:
                break
    return torch.tensor(cur, dtype=torch.int16), zc, z0


def main():
    method, split = sys.argv[1], sys.argv[2]
    names = P.load_cache(method, split)["names"]
    a, b = int(sys.argv[3]), int(sys.argv[4])
    out = OUT / f"{split}_{method}_win_{a}.pt"
    print("n boxes", len(BOX), flush=True)
    if out.exists():
        print("exists", out); print("---\nval_miou: 0.000000"); return
    wm = load_watermark(method, P.TASK_ROOT / "models", "cuda")
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {q["image_name"]: i for i, q in enumerate(ds.meta)}
    rng = np.random.RandomState(41 + a)
    rec = {}; t0 = time.time()
    for n in names[a:b]:
        it = ds[pos[n]]
        x = torch.from_numpy(np.ascontiguousarray(it["image"])).permute(2, 0, 1).float().div(255).cuda()
        m = torch.from_numpy(np.asarray(it["message"]).astype(np.float32)).flatten()
        mc = torch.from_numpy(rng.randint(0, 2, m.shape).astype(np.float32))
        with torch.no_grad():
            lg = crop_logits(wm, x)
            for suf, msg in (("", m), ("_ctrl", mc)):
                s1, s2 = stats(lg, msg)
                rb, rz, rz0 = refine(wm, x, msg, starts(s1 / s2.clamp(min=1e-6).sqrt()))
                for k, v in (("s1", s1.half()), ("s2", s2.half()), ("rbox", rb), ("rz", rz), ("rz0", rz0)):
                    rec.setdefault(k + suf, []).append(v)
    rec = {k: torch.stack(v) for k, v in rec.items()}; rec["names"] = names[a:b]; rec["shapes"] = SHAPES
    rec["box"] = torch.tensor([b_[:4] for b_ in BOX], dtype=torch.int16); rec["shape_idx"] = torch.tensor([b_[4] for b_ in BOX])
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{method} {split} [{a}:{b}] {time.time() - t0:.0f}s -> {out} | median rz {float(rec['rz'].max(1).values.median()):.2f} "
          f"ctrl {float(rec['rz_ctrl'].max(1).values.median()):.2f} | median gain {float((rec['rz'] - rec['rz0']).median()):.2f}", flush=True)
    print("---\nval_miou: 0.000000")


if __name__ == "__main__":
    main()
