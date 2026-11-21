"""N03b: finer message attribution for PixelSeal / TrustMark (extension of N03's success on StegaStamp).
A(x) = sum_k s_k logit_k(x), s = 2m - 1 (true message) or a random message (control).
attr_l = sum_c (dA/dh_l * h_l) at intermediate layers h_l (gradient x activation). For TrustMark (ResNet-50, BN in
eval, ReLU) the network is piecewise linear, so with gates fixed this is an exact decomposition of A up to bias
terms; for PixelSeal (ConvNeXt: GELU, LayerNorm) it is a first-order attribution.
Layers: TrustMark layer2 (31x31), layer3 (16x16); PixelSeal: the two ConvNeXt stage outputs with spatial size 16
and 32 at the wrapper's 256-px input (found by hooks). Extractor weights frozen; no shift-and-stitch.
Writes <research_cache>/N03b/{split}_{method}_gxa.pt {"names", "true": {layer: [N,1,h,w]}, "ctrl": {...}} and
prints tie-aware per-image AUROC on train (true vs control)."""
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
from scipy.stats import rankdata  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/N03b"); OUT.mkdir(parents=True, exist_ok=True)


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


class Tap:
    """Forward hooks keeping selected activations with grad."""
    def __init__(self, mods):
        self.acts = {}; self.h = [m.register_forward_hook(self._mk(n)) for n, m in mods.items()]

    def _mk(self, n):
        def f(_, __, out):
            out = out[0] if isinstance(out, (tuple, list)) else out
            out.retain_grad(); self.acts[n] = out
        return f


def setup(method):
    wm = load_watermark(method, P.TASK_ROOT / "models", "cuda")
    if method == "trustmark":
        net = wm.net; mods = {"layer2": net.layer2, "layer3": net.layer3}
        fwd = lambda x: net(wm._prep(x)).float()
        params = net.parameters()
    else:
        det = wm.det; cand = {}
        hooks = []
        x0 = torch.zeros(1, 3, 512, 512, device="cuda")
        shapes = {}
        for n, mm in det.convnext.named_modules():
            if n.count(".") == 1 and ("stages" in n or "stage" in n):
                hooks.append(mm.register_forward_hook(lambda m_, i_, o_, n=n: shapes.__setitem__(n, tuple((o_[0] if isinstance(o_, (tuple, list)) else o_).shape))))
        with torch.no_grad():
            det(wm._resize(x0))
        for h in hooks:
            h.remove()
        print("PixelSeal convnext stage outputs:", shapes)
        mods = {}
        for n, s in shapes.items():
            if len(s) == 4 and s[-1] in (16, 32):
                mods[f"stage{s[-1]}"] = dict(det.convnext.named_modules())[n]
        fwd = lambda x: det(wm._resize(x))[:, 1:].float()
        params = det.parameters()
    for p in params:
        p.requires_grad_(False)
    return wm, mods, fwd


for method in ("trustmark", "pixelseal"):
    wm, mods, fwd = setup(method)
    print(method, "layers:", list(mods))
    for split in ("train", "test"):
        out = OUT / f"{split}_{method}_gxa.pt"
        if out.exists():
            print("exists", out); continue
        ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
        names = P.load_cache(method, split)["names"]; rng = np.random.RandomState(11)
        res = {"true": {k: [] for k in mods}, "ctrl": {k: [] for k in mods}}; t0 = time.time()
        for s in range(0, len(names), 16):
            items = [ds[pos[n]] for n in names[s:s + 16]]
            x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255).cuda()
            sg = 2 * torch.from_numpy(np.stack([it["message"] for it in items])).float().cuda() - 1
            ctrl = 2 * torch.from_numpy(rng.randint(0, 2, sg.shape)).float().cuda() - 1
            for key, sv in (("true", sg), ("ctrl", ctrl)):
                tap = Tap(mods)
                with torch.enable_grad():
                    lg = fwd(x.clone().requires_grad_(True)); A = (lg * sv).sum()
                    A.backward()
                for k, a in tap.acts.items():
                    res[key][k].append((a.grad * a).sum(1, keepdim=True).detach().float().cpu())
                for h in tap.h:
                    h.remove()
        rec = {"names": names, **{key: {k: torch.cat(v) for k, v in d.items()} for key, d in res.items()}}
        tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
        print(f"{method} {split}: {len(names)} in {time.time() - t0:.0f}s " +
              ", ".join(f"{k} {tuple(v.shape)}" for k, v in rec["true"].items()), flush=True)
        if split == "train":
            masks = P.load_cache(method)["masks"].numpy()
            for key in ("true", "ctrl"):
                line = []
                for k, v in rec[key].items():
                    up = F.interpolate(v.float(), size=128, mode="bilinear", align_corners=False)[:, 0].numpy()
                    line.append(f"{k} {np.nanmean([auroc(up[i], masks[i]) for i in range(len(masks))]):.4f}")
                print(f"  AUROC {method} {key}: " + ", ".join(line), flush=True)
    del wm; torch.cuda.empty_cache()
print("---\nval_miou: 0.000000\nN03b GXA DONE")
