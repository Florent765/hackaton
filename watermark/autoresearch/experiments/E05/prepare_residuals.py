"""E05: native multiband residuals from the 512 RGB image (before any pooling).
Luminance Y = 0.299R + 0.587G + 0.114B in [0,1]; responses: Y - G_sigma(Y) for sigma = 1, 2, 4 px (reflect pad),
horizontal / vertical central derivatives. Each response summarised on 4x4 blocks by signed mean and RMS
-> 10 channels on the official 128 grid. Fixed transform sign(x)*log1p(|x|*100) (no learned statistics -> no fold
leakage). Writes <research_cache>/E05/{split}_{method}_res.pt {"names", "res": [N,10,128,128] float16, "channels"}."""
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

OUT = Path("/p/project1/training2666/jacquet2/research_cache/E05"); OUT.mkdir(parents=True, exist_ok=True)


def gblur(y, s):
    k = int(3 * s) * 2 + 1; t = torch.arange(k, device=y.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    y = F.conv2d(F.pad(y, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(y, g.view(1, 1, k, 1))


def responses(x):
    y = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    r = [y - gblur(y, s) for s in (1.0, 2.0, 4.0)]
    yp = F.pad(y, (1, 1, 1, 1), mode="reflect")
    r.append((yp[:, :, 1:-1, 2:] - yp[:, :, 1:-1, :-2]) / 2)
    r.append((yp[:, :, 2:, 1:-1] - yp[:, :, :-2, 1:-1]) / 2)
    out = []
    for v in r:
        out.append(F.avg_pool2d(v, 4)); out.append(F.avg_pool2d(v * v, 4).sqrt())
    o = torch.cat(out, 1)
    return torch.sign(o) * torch.log1p(o.abs() * 100)


CH = [f"{n}_{s}" for n in ("hp1", "hp2", "hp4", "dx", "dy") for s in ("mean", "rms")]
for split in ("train", "test"):
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    for method in P.METHODS:
        out = OUT / f"{split}_{method}_res.pt"
        if out.exists():
            print("exists", out); continue
        names = P.load_cache(method, split)["names"]; res = []; t0 = time.time()
        for s in range(0, len(names), 32):
            x = torch.from_numpy(np.stack([ds[pos[n]]["image"] for n in names[s:s + 32]])).permute(0, 3, 1, 2).float().div(255).cuda()
            res.append(responses(x).half().cpu())
        rec = {"names": names, "res": torch.cat(res), "channels": CH}
        assert torch.isfinite(rec["res"].float()).all()
        tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
        print(f"{split} {method}: {tuple(rec['res'].shape)} in {time.time() - t0:.0f}s", flush=True)
print("---\nval_miou: 0.000000\nE05 RES DONE")
