"""Shared unlearning library (exact starter contract). Import from any branch script:
    import sys; sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); import ul_lib as L
Facts (starter task_template.py): timm vit_base_patch16_224, num_classes=100, init_values=1e-4, pretrained=False;
images uint8 [N,224,224,3] -> x = (pixels/255 - 0.5)/0.5; server = eval + fp16 autocast; export float32 strict state dict.
No labels / retain / gold: everything here is a DIAGNOSTIC (pseudo-labels = original argmax, never true labels)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

U = Path(os.environ.get("UL_ROOT", "/p/scratch/training2666/delorme2/team51/unlearning"))
HOME_OUT = Path("/p/home/jusers/jacquet2/jureca/ul_out")          # exports here (login node can upload them)
CACHE = Path(os.environ.get("UL_OUT", "/p/project1/training2666/jacquet2/ul")) / "cache"
DEV = "cuda"


def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)


def load_original():
    m = build_model(); sd = torch.load(U / "model.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True); return m.to(DEV).eval()


def original_state():
    return torch.load(U / "model.pt", map_location="cpu", weights_only=True)


def load_images():
    with np.load(U / "forget_images.npz", allow_pickle=False) as d:
        return torch.from_numpy(d["images"].copy())                     # uint8 [N,224,224,3]


def prep(u8):
    return (u8.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


# ------------------------------------------------------------------ views (deterministic, seedable)
def view(x, kind, seed=0):
    """x: preprocessed [B,3,224,224] in [-1,1]. Families: 'clean', 'mild' (flip+small shift+noise), 'strong'
    (crop-resize+color+noise), 'diag' (rotation+blur: reserved for diagnostics, never for training)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    if kind == "clean":
        return x
    if kind == "mild":
        y = torch.flip(x, dims=[3]) if seed % 2 else x
        dx, dy = (torch.randint(-8, 9, (2,), generator=g, device=x.device)).tolist()
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
        return (y + 0.03 * torch.randn(y.shape, generator=g, device=x.device)).clamp(-1, 1)
    if kind == "strong":
        s = 0.6 + 0.3 * torch.rand(1, generator=g, device=x.device).item()
        h = int(224 * s); i, j = [int(v) for v in torch.randint(0, 224 - h + 1, (2,), generator=g, device=x.device)]
        y = F.interpolate(x[:, :, i:i + h, j:j + h], size=224, mode="bilinear", align_corners=False)
        c = 1 + 0.3 * (torch.rand(1, 3, 1, 1, generator=g, device=x.device) - 0.5)
        y = y * c + 0.1 * (torch.rand(1, 3, 1, 1, generator=g, device=x.device) - 0.5)
        return (y + 0.05 * torch.randn(y.shape, generator=g, device=x.device)).clamp(-1, 1)
    if kind == "diag":
        ang = (torch.rand(1, generator=g, device=x.device).item() - 0.5) * 0.5
        th = torch.tensor([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0]], device=x.device, dtype=x.dtype)
        y = F.grid_sample(x, F.affine_grid(th.expand(len(x), 2, 3), list(x.shape), align_corners=False), align_corners=False,
                          padding_mode="reflection")
        k = torch.tensor([1, 4, 6, 4, 1], device=x.device, dtype=x.dtype); k = (k[:, None] * k[None]) / 256
        return F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
    raise ValueError(kind)


@torch.no_grad()
def logits(model, u8, kind="clean", seed=0, bs=128, idx=None):
    """Server-like logits (eval + fp16 autocast) on a view family."""
    model.eval(); out = []; idx = range(len(u8)) if idx is None else idx; idx = list(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(view(prep(u8[b]), kind, seed + s)).float().cpu())
    return torch.cat(out)


def teacher_cache(u8, kinds=("clean", "mild", "strong", "diag"), seeds=(0,)):
    """Original-model logits per view family, cached on project1 (computed once)."""
    CACHE.mkdir(parents=True, exist_ok=True); p = CACHE / "orig_logits.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    m = load_original(); rec = {}
    for k in kinds:
        for s in seeds:
            rec[f"{k}_{s}"] = logits(m, u8, k, s)
    torch.save(rec, p); return rec


# ------------------------------------------------------------------ diagnostics (NOT the official score)
def diag(lg, lg_orig, T=2.0):
    p = F.log_softmax(lg / T, 1); q = F.log_softmax(lg_orig / T, 1)
    kl_orig_sub = (q.exp() * (q - p)).sum(1)                                   # KL(original || sub), T=2
    pl = lg_orig.argmax(1); ce = F.cross_entropy(lg, pl, reduction="none")
    ent = -(F.softmax(lg, 1) * F.log_softmax(lg, 1)).sum(1); conf = F.softmax(lg, 1).max(1).values
    q_ = lambda v: [round(float(x), 4) for x in torch.quantile(v.float(), torch.tensor([0.05, 0.5, 0.95]))]
    return {"kl_orig_sub_T2": round(float(kl_orig_sub.mean()), 5), "ce_pseudo_mean": round(float(ce.mean()), 4),
            "ce_pseudo_std": round(float(ce.std()), 4), "ce_q05_50_95": q_(ce), "conf_mean": round(float(conf.mean()), 4),
            "entropy_mean": round(float(ent.mean()), 4), "argmax_change": round(float((lg.argmax(1) != pl).float().mean()), 4),
            "n_pred_classes": int(lg.argmax(1).unique().numel())}


def full_diag(model, u8, tc, idx=None):
    idx = list(range(len(u8))) if idx is None else list(idx)
    return {k: diag(logits(model, u8, k, 0, idx=idx), tc[f"{k}_0"][idx]) for k in ("clean", "mild", "strong", "diag")}


# ------------------------------------------------------------------ export
def interpolate(sd0, sd1, alpha):
    return {k: (sd0[k].float() + alpha * (sd1[k].float() - sd0[k].float())) if sd0[k].is_floating_point() else sd1[k]
            for k in sd0}


def export(sd, name, meta=None):
    """float32, finite, strict reload into build_model, size <= 450 MB, fp16-autocast forward finite. Returns path."""
    HOME_OUT.mkdir(parents=True, exist_ok=True); path = HOME_OUT / f"{name}.pt"
    sd = {k: v.detach().cpu().float().contiguous() for k, v in sd.items()}
    assert all(torch.isfinite(v).all() for v in sd.values()), "non-finite weights"
    torch.save(sd, path)
    m = build_model(); m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    m = m.to(DEV).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        y = m(torch.zeros(2, 3, 224, 224, device=DEV))
    assert torch.isfinite(y).all(), "non-finite fp16 output"
    size = path.stat().st_size / 1e6; assert size <= 450, size
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    (HOME_OUT / f"{name}.json").write_text(json.dumps({"name": name, "sha16": h, "size_mb": round(size, 1),
                                                         "time": time.ctime(), **(meta or {})}, indent=1, default=str))
    print(f"EXPORT {path} {size:.1f} MB sha16 {h} (strict reload + fp16 forward OK)", flush=True)
    return path


def report(name, d):
    print("DIAG " + name + " " + json.dumps(d), flush=True)
