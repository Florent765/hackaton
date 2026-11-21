"""A03 probe (agent idea N02, differential form): message-specific matched filter from re-embedding.
For an image x with known message m and a random message m', d = embed(x, m) - embed(x, m') is the message-dependent
part of the residual the embedder would add (content-adaptive parts cancel). If a region of x still carries the
original watermark r(x0, m) ~ r(x, m), then <HP(x), d> > 0 there and ~0 elsewhere.
Map: per 16x16 px block (512 frame -> 32x32): sum_c <HP(x), d> / sqrt(sum HP(x)^2 * sum d^2).
Control: same with d_ctrl = embed(x, m'') - embed(x, m') (two random messages).
Prints per-method tie-aware per-image AUROC vs the 128 mask (all / lo / hi regime) on the first NIMG train images.
usage: probe_resid.py [nimg]"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from scipy.stats import rankdata  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

NIMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def emb_float(name, wm, img, bits):
    """float embedded image (0..255) without the final rounding when the API allows it."""
    if name == "pixelseal":
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).cuda()
        msgs = torch.tensor([list(map(int, bits))], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out = wm.model.embed(x, msgs=msgs, is_video=False)
        return out["imgs_w"][0].permute(1, 2, 0).double().cpu().numpy() * 255
    if name == "stegastamp":
        h, w = img.shape[:2]
        small = np.asarray(Image.fromarray(img).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255
        res = wm._run(small[None], ["residual"], np.asarray([bits], np.float32))[0][0] * 255
        res = res - res.mean(axis=(0, 1), keepdims=True)
        full = np.stack([np.asarray(Image.fromarray(res[:, :, c], mode="F").resize((w, h), Image.Resampling.BILINEAR))
                         for c in range(3)], 2)
        return img.astype(np.float64) + full
    return wm.embed(img, bits).astype(np.float64)


def hp(x):
    t = torch.from_numpy(x).permute(2, 0, 1)[None].float()
    k = torch.tensor([1., 4., 6., 4., 1.]); k = (k[:, None] * k[None]) / 256
    bl = F.conv2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), k[None, None].repeat(3, 1, 1, 1), groups=3)
    return t - bl


def corr_map(hx, d):
    num = F.avg_pool2d((hx * d).sum(1, keepdim=True), 16)
    den = (F.avg_pool2d((hx ** 2).sum(1, keepdim=True), 16) * F.avg_pool2d((d ** 2).sum(1, keepdim=True), 16)).sqrt()
    return (num / den.clamp(min=1e-9))[0, 0]


ds = LocalisationDataset(P.TASK_ROOT / "release", "train")
rng = np.random.RandomState(3)
for name in ("pixelseal", "trustmark", "stegastamp"):
    t0 = time.time()
    wm = load_watermark(name, P.TASK_ROOT / "models", "cuda")
    c = P.load_cache(name); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    hi = (c["feats"]["scalars"][:, 1] > R.BACC_SPLIT).numpy()
    res = {"true": [], "ctrl": [], "true_raw": []}; regs = []
    for i, n in enumerate(c["names"][:NIMG]):
        it = ds[pos[n]]; img = it["image"]; m = np.asarray(it["message"]).astype(int)
        m1 = rng.randint(0, 2, m.shape); m2 = rng.randint(0, 2, m.shape)
        e0 = emb_float(name, wm, img, m); e1 = emb_float(name, wm, img, m1); e2 = emb_float(name, wm, img, m2)
        hx = hp(img.astype(np.float64))
        dt = torch.from_numpy(e0 - e1).permute(2, 0, 1)[None].float(); dc = torch.from_numpy(e2 - e1).permute(2, 0, 1)[None].float()
        mask = c["masks"][i].numpy()
        for key, d in (("true", dt), ("ctrl", dc), ("true_raw", torch.from_numpy(e0 - img).permute(2, 0, 1)[None].float())):
            mp = corr_map(hx, d)
            up = F.interpolate(mp[None, None], size=128, mode="bilinear", align_corners=False)[0, 0].numpy()
            res[key].append(auroc(up, mask))
        regs.append(hi[i])
        if i == 0:
            print(name, "residual rms true-vs-rand diff", float(dt.pow(2).mean().sqrt()), "raw residual rms", float(np.sqrt(((e0 - img) ** 2).mean())), flush=True)
    regs = np.array(regs)
    for key, v in res.items():
        v = np.array(v)
        print(f"{name} AUROC {key}: all {np.nanmean(v):.4f} lo {np.nanmean(v[~regs]):.4f} hi {np.nanmean(v[regs]):.4f} (n={len(v)})", flush=True)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    del wm; torch.cuda.empty_cache()
print("---\nval_miou: 0.000000\nA03 PROBE DONE")
