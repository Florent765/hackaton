"""A04 probe: multi-scale crop decoding (decoder as a sliding verifier).
Hypothesis: in the low-coverage regime the watermark sits on a rectangle that was embedded at its own scale; a crop
that matches it, resized to the decoder resolution, decodes the known message far better than the whole image.
For square windows s in SIZES (stride s/4) we decode every crop and compute the bit accuracy with the true message
(and with a random message: control). Reported on the first NIMG lo-flag (bit acc <= 0.75) train images per method:
  full-image BA, max crop BA (true / ctrl), IoU of the best crop box with the GT mask, IoU of the GT bbox,
  per-image AUROC of the map 'max BA over crops covering the cell' (true / ctrl).
usage: probe_crops.py [nimg]"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from scipy.stats import rankdata  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

NIMG = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SIZES = (96, 128, 176, 256, 352)


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def boxes():
    out = []
    for s in SIZES:
        st = s // 4; pos = list(range(0, 512 - s + 1, st))
        if pos[-1] != 512 - s:
            pos.append(512 - s)
        out += [(y, x, s) for y in pos for x in pos]
    return out


BOX = boxes()


def crop_logits(wm, x):
    """x [3,512,512] float cuda -> logits [n_boxes, nbits]"""
    out = []
    for s in SIZES:
        bs = [b for b in BOX if b[2] == s]
        crops = torch.stack([x[:, y:y + s, xx:xx + s] for y, xx, _ in bs])
        for i in range(0, len(crops), 128):
            out.append(wm.logits(crops[i:i + 128]).float().cpu())
    return torch.cat(out)


ds = LocalisationDataset(P.TASK_ROOT / "release", "train")
rng = np.random.RandomState(5)
print("n boxes", len(BOX), flush=True)
for name in ("pixelseal", "trustmark", "stegastamp"):
    t0 = time.time()
    wm = load_watermark(name, P.TASK_ROOT / "models", "cuda")
    c = P.load_cache(name); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    hi = (c["feats"]["scalars"][:, 1] > R.BACC_SPLIT).numpy()
    sel = np.nonzero(~hi)[0][:NIMG]
    st = {k: [] for k in ("full", "max_t", "max_c", "iou_best", "iou_bbox", "auc_t", "auc_c")}
    for i in sel:
        it = ds[pos[c["names"][i]]]
        x = torch.from_numpy(it["image"]).permute(2, 0, 1).float().div(255).cuda()
        m = torch.from_numpy(np.asarray(it["message"]).astype(np.float32))
        mc = torch.from_numpy(rng.randint(0, 2, m.shape).astype(np.float32))
        full = (wm.logits(x[None]).float().cpu()[0] > 0).float()
        st["full"].append(float((full == m).float().mean()))
        lg = crop_logits(wm, x); bits = (lg > 0).float()
        mask = c["masks"][i].numpy().astype(bool)
        ys, xs = np.nonzero(mask); gb = np.zeros_like(mask); gb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        st["iou_bbox"].append((gb & mask).sum() / (gb | mask).sum())
        for key, msg in (("t", m), ("c", mc)):
            ba = (bits == msg[None]).float().mean(1).numpy()
            st[f"max_{key}"].append(float(ba.max()))
            mp = np.zeros((128, 128), np.float32)
            for (y, xx, s), b in zip(BOX, ba):
                sl = (slice(y // 4, (y + s) // 4), slice(xx // 4, (xx + s) // 4))
                mp[sl] = np.maximum(mp[sl], b)
            st[f"auc_{key}"].append(auroc(mp, mask))
            if key == "t":
                y, xx, s = BOX[int(ba.argmax())]
                pb = np.zeros_like(mask); pb[y // 4:(y + s) // 4, xx // 4:(xx + s) // 4] = True
                st["iou_best"].append((pb & mask).sum() / (pb | mask).sum())
    print(name, f"n={len(sel)} {time.time() - t0:.0f}s", {k: round(float(np.nanmean(v)), 4) for k, v in st.items()}, flush=True)
    print(name, "max_t per image:", np.round(st["max_t"], 2).tolist(), flush=True)
    print(name, "iou_best per image:", np.round(st["iou_best"], 2).tolist(), flush=True)
    del wm; torch.cuda.empty_cache()
print("---\nval_miou: 0.000000\nA04 PROBE DONE")
