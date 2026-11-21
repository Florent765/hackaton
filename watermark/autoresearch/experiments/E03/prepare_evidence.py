"""E03: per-bit message evidence maps for PixelSeal and TrustMark (StegaStamp has no local logits).
For dense (shift-and-stitch x4, as REF) per-position logits l_k(h,w) and signs s_k = 2 m_k - 1, r_k = s_k l_k:
  new channels (not already in REF's agree = [mean_k r_k, mean_k s_k sign(l_k)]):
    std_k r_k,  q10_k r_k,  Z = sum_k r_k / sqrt(sum_k l_k^2 + eps)
  control: the same three channels with a fixed random message (same K) -> diagnosis only.
Writes <research_cache>/E03/{split}_{method}_evidence.pt = {"names", "true": [N,3,32,32], "ctrl": [N,3,32,32]}
and prints tie-aware per-image AUROC (train) of each channel, true vs control."""
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
import baseline as B  # noqa: E402  (fixed extractor code, same densification as the cache)
from scipy.stats import rankdata  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/E03"); OUT.mkdir(parents=True, exist_ok=True)


def stats(pp, sign):
    r = pp * sign[:, :, None, None]
    z = r.sum(1) / torch.sqrt((pp ** 2).sum(1) + 1e-6)
    return torch.stack([r.std(1), torch.quantile(r.flatten(2).transpose(1, 2).reshape(-1, r.shape[1]), 0.1, dim=1)
                        .view(r.shape[0], r.shape[2], r.shape[3]), z], 1)


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


for method in ("pixelseal", "trustmark"):
    ex = B.FrozenExtractor(method, load_watermark, P.TASK_ROOT / "models", Path("/tmp"), "cuda", shifts=4)
    for split in ("train", "test"):
        out = OUT / f"{split}_{method}_evidence.pt"
        if out.exists():
            print("exists", out); continue
        ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
        names = P.load_cache(method, split)["names"]
        rng = np.random.RandomState(123); t0 = time.time(); T, C = [], []
        for s in range(0, len(names), 16):
            items = [ds[pos[n]] for n in names[s:s + 16]]
            x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255).cuda()
            msg = torch.from_numpy(np.stack([it["message"] for it in items])).float().cuda()
            ctrl = torch.from_numpy(rng.randint(0, 2, msg.shape)).float().cuda()
            with torch.no_grad():
                pp = ex._dense_features(x)["per_position_logits"].float()
                T.append(stats(pp, 2 * msg - 1).half().cpu()); C.append(stats(pp, 2 * ctrl - 1).half().cpu())
        rec = {"names": names, "true": torch.cat(T), "ctrl": torch.cat(C), "channels": ["std", "q10", "Z"]}
        tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
        print(f"{method} {split}: {len(names)} in {time.time() - t0:.0f}s, shape {tuple(rec['true'].shape)}", flush=True)
        if split == "train":
            masks = P.load_cache(method)["masks"].numpy()
            for key in ("true", "ctrl"):
                a = []
                for ch, nm in enumerate(rec["channels"]):
                    up = F.interpolate(rec[key][:, ch:ch + 1].float(), size=128, mode="bilinear", align_corners=False)[:, 0].numpy()
                    a.append(f"{nm} {np.nanmean([auroc(up[i], masks[i]) for i in range(len(masks))]):.4f}")
                print(f"  AUROC {method} {key}: " + ", ".join(a), flush=True)
    del ex; torch.cuda.empty_cache()
print("---\nval_miou: 0.000000\nE03 EVIDENCE DONE")
