"""Watermark localisation baseline: frozen organiser extractors + small per-method segmentation head.

Modes
  smoke     tiny end-to-end check (load -> features -> logits -> loss -> backward, gradients, metric, masks)
  train     train one head per method, keep the best checkpoint by official validation mIoU
  evaluate  validation metrics (overall, per method, per group), constant baselines, overlays
  predict   test inference and a validated submission .pt

Run from the task root (the folder holding wmloc/, models/, release/), e.g.
  python work/<user>/<run>/baseline.py smoke --methods pixelseal
See README_baseline.md next to this file for the Slurm commands.

Inputs of the head, per method (all from the organiser wrappers in wmloc/watermarks.py, extractor frozen in eval):
  pixelseal   features() maps "embedding" and "convnext"; per-position message agreement from "per_position_logits"
  trustmark   features() maps "layer2", "layer3"; per-position message agreement from "per_position_logits"
  stegastamp  the wrapper exposes only global logits; we add intermediate decoder conv outputs to a *copy* of the
              ONNX graph (dec_conv_k/Relu). No per-position logits exist (dense layers after flatten), so no
              agreement map for this method.
  all         RGB image average-pooled to the head grid, and two image-level scalars from the *global* logits and
              the known message (soft agreement, bit accuracy), broadcast as constant channels.
The per-position agreement is sum_k (2 m_k - 1) * l_k(h, w) / nbits, where l(h, w) are the decoder's own logits
before its global mean pooling (the global logit is their spatial mean). No bit-to-pixel mapping is assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("pixelseal", "trustmark", "stegastamp")
FEATURES = {
    "pixelseal": ("embedding", "convnext"),
    "trustmark": ("layer2", "layer3"),
    "stegastamp": ("gen_decoder/dec_conv_3/Relu:0", "gen_decoder/dec_conv_5/Relu:0"),
}
STEGA_ALL = tuple(f"gen_decoder/dec_conv_{k}/Relu:0" for k in range(1, 8))
GRID = 128            # official mask grid
HEAD_GRID = 64        # internal head resolution, upsampled to GRID for the logits
THRESHOLDS = tuple(np.round(np.arange(0.20, 0.81, 0.05), 2).tolist())


# --------------------------------------------------------------------------------------------- utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def import_starter(task_root: Path, extra_paths=()):
    sys.path.insert(0, str(task_root))
    for p in extra_paths:                                  # e.g. private trustmark copy with its weights
        if p and Path(p).exists():
            sys.path.insert(0, str(Path(p).resolve()))
    from wmloc import utils as wutils                      # noqa: E402
    from wmloc.data import LocalisationDataset            # noqa: E402
    from wmloc.watermarks import load_watermark           # noqa: E402
    return wutils, LocalisationDataset, load_watermark


def miou_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Vectorised official per-image mIoU over classes {0,1}; empty union -> 1. pred, gt: [B,H,W] in {0,1}."""
    p = pred.bool().flatten(1); g = gt.bool().flatten(1)
    out = []
    for pc, gc in ((~p, ~g), (p, g)):
        inter = (pc & gc).sum(1).float(); union = (pc | gc).sum(1).float()
        out.append(torch.where(union == 0, torch.ones_like(union), inter / union.clamp(min=1)))
    return (out[0] + out[1]) / 2


# --------------------------------------------------------------------------------------------- features
class FrozenExtractor:
    """Wraps the organiser wrapper of one method and returns a dict of spatial maps + scalars for a batch."""

    def __init__(self, method, load_watermark, models_dir: Path, work_models: Path, device: str, shifts: int = 1):
        self.method, self.device, self.shifts = method, device, shifts
        self.wm = load_watermark(method, models_dir, device)
        self.torch_modules = []
        if method == "pixelseal":
            self.torch_modules = [self.wm.model]
        elif method == "trustmark":
            self.torch_modules = [self.wm.net]
        else:
            import onnxruntime as ort
            path = ensure_stega_onnx(models_dir, work_models)
            prov = ([("CUDAExecutionProvider", {"device_id": torch.device(device).index or 0})]
                    if "cuda" in str(device) else []) + ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(path), providers=prov)
            self.providers = self.sess.get_providers()
        for m in self.torch_modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _dense_features(self, x: torch.Tensor) -> dict:
        """Shift-and-stitch: the extractor's last maps are 8x8 (stride 32 at its working resolution). Run it on
        S x S circularly shifted copies of the 512 input (shift = one 1/S sub-cell, rounded to whole 512 pixels)
        and interleave the outputs into an (8S x 8S) map. Wrap-around only affects the last sub-cells of the
        bottom/right border. The wrapper itself (resize, preprocessing) is used unchanged."""
        base = self.wm.features(x)
        S = self.shifts
        h, w = base["per_position_logits"].shape[-2:]
        keys = [k for k in (*FEATURES[self.method], "per_position_logits") if base[k].shape[-2:] == (h, w)]
        if S <= 1:
            return base
        out = {k: base[k].new_zeros(base[k].shape[0], base[k].shape[1], h * S, w * S) for k in keys}
        step_y, step_x = x.shape[-2] / (h * S), x.shape[-1] / (w * S)
        for a in range(S):
            for b in range(S):
                dy, dx = int(round(a * step_y)), int(round(b * step_x))
                f = base if a == b == 0 else self.wm.features(torch.roll(x, shifts=(-dy, -dx), dims=(2, 3)))
                for k in keys:
                    out[k][:, :, a::S, b::S] = f[k]
        return {**base, **out}

    @torch.no_grad()
    def __call__(self, images_u8: torch.Tensor, messages: torch.Tensor) -> dict:
        """images_u8: [B,512,512,3] uint8 (CPU); messages: [B,nbits] {0,1}. Returns float32 CPU tensors."""
        x = images_u8.permute(0, 3, 1, 2).float().div(255)
        sign = (2 * messages.float() - 1)
        out = {}
        if self.method == "stegastamp":
            names = list(FEATURES["stegastamp"]) + [self.wm.LOGITS]
            res = self.sess.run(names, {"image": self.wm._prep(x).astype(np.float32),
                                        "secret": np.zeros((len(x), 100), np.float32)})
            for n, r in zip(names[:-1], res[:-1]):
                out[n] = torch.from_numpy(r).float()
            glog = torch.from_numpy(res[-1]).float()
        else:
            f = self._dense_features(x.to(self.device))
            for n in FEATURES[self.method]:
                out[n] = f[n].float().cpu()
            pp = f["per_position_logits"].float()                               # [B,nbits,h,w]
            s = sign.to(pp.device)[:, :, None, None]
            out["agree"] = torch.cat([(pp * s).mean(1, keepdim=True),          # logit-weighted agreement
                                      ((pp > 0).float() * 2 - 1).mul(s).mean(1, keepdim=True)], 1).cpu()
            glog = self.wm.logits(x.to(self.device)).float().cpu()
        soft = (glog * sign).mean(1) / glog.abs().mean(1).clamp(min=1e-6)       # in [-1,1]
        bacc = ((glog > 0).float() == messages.float()).float().mean(1)
        out["scalars"] = torch.stack([soft, bacc], 1)                           # [B,2]
        out["rgb"] = F.adaptive_avg_pool2d(x, HEAD_GRID)                         # [B,3,64,64]
        return out


def ensure_stega_onnx(models_dir: Path, work_models: Path) -> Path:
    """Copy of the organiser StegaStamp graph with intermediate decoder conv outputs exposed (original untouched)."""
    dst = work_models / "stega_stamp_feats.onnx"
    if dst.exists():
        return dst
    import onnx
    m = onnx.load(str(models_dir / "stegastamp" / "stega_stamp_logits.onnx"))
    have = {o.name for o in m.graph.output}
    for name in STEGA_ALL:
        if name not in have:
            vi = onnx.helper.make_empty_tensor_value_info(name)
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            m.graph.output.append(vi)
    work_models.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp"); onnx.save(m, str(tmp)); os.replace(tmp, dst)
    return dst


def extract_all(extractor, ds, indices, batch_size, half=True):
    """Run the frozen extractor over ds[indices]; returns (feature dict of stacked CPU tensors, items meta)."""
    feats, meta = {}, []
    for s in range(0, len(indices), batch_size):
        items = [ds[i] for i in indices[s:s + batch_size]]
        imgs = torch.from_numpy(np.stack([it["image"] for it in items]))
        msgs = torch.from_numpy(np.stack([it["message"] for it in items]))
        f = extractor(imgs, msgs)
        for k, v in f.items():
            feats.setdefault(k, []).append(v.half() if (half and v.dim() == 4) else v)
        for it in items:
            meta.append({"image_name": it["image_name"], "method": it["method"],
                         "mask128": it.get("mask128"), "mask": it.get("mask"),
                         "full_watermark": it.get("full_watermark")})
    return {k: torch.cat(v) for k, v in feats.items()}, meta


# --------------------------------------------------------------------------------------------- head
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegHead(nn.Module):
    """Per-map 1x1 projections -> upsample to 64x64 -> concat (+agreement, rgb, scalars) -> dilated convs
    -> upsample to 128 -> 1 logit channel."""

    def __init__(self, map_channels: dict, n_agree: int, proj=32, width=96):
        super().__init__()
        self.map_names = list(map_channels)
        self.proj = nn.ModuleDict({self._k(n): ConvBNReLU(c, proj, k=1) for n, c in map_channels.items()})
        cin = proj * len(map_channels) + n_agree + 3 + 2
        self.body = nn.Sequential(ConvBNReLU(cin, width), ConvBNReLU(width, width, d=2),
                                  ConvBNReLU(width, width, d=4), ConvBNReLU(width, width, d=8),
                                  ConvBNReLU(width, width))
        self.up = nn.Sequential(ConvBNReLU(width + 3, 32), nn.Conv2d(32, 1, 1))

    @staticmethod
    def _k(n):
        return n.replace("/", "_").replace(":", "_")

    def forward(self, f: dict) -> torch.Tensor:
        parts = [F.interpolate(self.proj[self._k(n)](f[n]), size=HEAD_GRID, mode="bilinear", align_corners=False)
                 for n in self.map_names]
        if "agree" in f:
            parts.append(F.interpolate(f["agree"], size=HEAD_GRID, mode="bilinear", align_corners=False))
        rgb = f["rgb"]
        parts += [rgb, f["scalars"][:, :, None, None].expand(-1, -1, HEAD_GRID, HEAD_GRID)]
        h = self.body(torch.cat(parts, 1))
        h = F.interpolate(h, size=GRID, mode="bilinear", align_corners=False)
        rgb128 = F.interpolate(rgb, size=GRID, mode="bilinear", align_corners=False)
        return self.up(torch.cat([h, rgb128], 1))[:, 0]                       # [B,128,128] logits


def loss_fn(logits, target, dice_weight=1.0, eps=1.0):
    """BCEWithLogits + per-image two-class soft Dice (mirrors the two-class per-image mIoU).
    Empty class: with eps smoothing, an empty target class predicted empty gives Dice ~1 (loss ~0) and any
    predicted mass on it is penalised, so empty masks need no special case."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
    dice = 0
    for pc, tc in ((p, t), (1 - p, 1 - t)):
        dice = dice + (2 * (pc * tc).sum(1) + eps) / (pc.sum(1) + tc.sum(1) + eps)
    return bce + dice_weight * (1 - dice.mean() / 2)


def to_device(f: dict, idx, device):
    return {k: v[idx].to(device, non_blocking=True).float() for k, v in f.items()}


@torch.no_grad()
def predict_probs(head, feats, device, batch_size=32, amp=True):
    head.eval(); out = []
    n = len(next(iter(feats.values())))
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(n, s + batch_size))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and "cuda" in str(device)):
            out.append(torch.sigmoid(head(to_device(feats, idx, device)).float()).cpu())
    return torch.cat(out)


def score_thresholds(probs, gts):
    """Official-metric mIoU per threshold; returns dict thr -> mean mIoU."""
    return {t: float(miou_batch(probs >= t, gts).mean()) for t in THRESHOLDS}


# --------------------------------------------------------------------------------------------- split
def make_split(meta, val_frac, seed):
    """Stratified by method x full_watermark. Image names are content hashes and every message is distinct, so
    source-image grouping is impossible from the release (variants of one source could straddle the split)."""
    rng = np.random.RandomState(seed); train, val = [], []
    strata = {}
    for i, m in enumerate(meta):
        strata.setdefault((m["method"], bool(m.get("full_watermark"))), []).append(i)
    for key in sorted(strata):
        idx = np.array(sorted(strata[key], key=lambda i: meta[i]["image_name"]))
        rng.shuffle(idx); nv = int(round(len(idx) * val_frac))
        val += idx[:nv].tolist(); train += idx[nv:].tolist()
    return sorted(train), sorted(val)


# --------------------------------------------------------------------------------------------- modes
def common_setup(args):
    task_root = Path(args.task_root).resolve()
    wutils, LocalisationDataset, load_watermark = import_starter(task_root, [args.pylib])
    if "trustmark" in args.methods or args.mode == "predict":
        import trustmark
        log("trustmark package:", trustmark.__file__)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        log("WARNING: running on CPU")
    return task_root, wutils, LocalisationDataset, load_watermark, device


def build_extractor(method, load_watermark, task_root, args, device):
    t0 = time.time()
    ex = FrozenExtractor(method, load_watermark, task_root / args.models_dir, Path(args.work_models), device,
                         shifts=args.shifts)
    log(f"[{method}] extractor loaded in {time.time() - t0:.1f}s")
    return ex


def mode_smoke(args):
    task_root, wutils, LDS, load_wm, device = common_setup(args)
    ds = LDS(task_root / args.data, "train")
    # 1) metric equivalence with the official implementation
    rng = np.random.RandomState(0)
    cases = [rng.randint(0, 2, (GRID, GRID)) for _ in range(3)] + [np.zeros((GRID, GRID), int), np.ones((GRID, GRID), int)]
    gts = cases[::-1]
    for p, g in zip(cases, gts):
        a = wutils.miou_image(p, g); b = float(miou_batch(torch.tensor(p)[None], torch.tensor(g)[None])[0])
        assert abs(a - b) < 1e-6, (a, b)
    assert float(miou_batch(torch.zeros(1, GRID, GRID), torch.zeros(1, GRID, GRID))[0]) == 1.0
    log("metric: vectorised mIoU matches wmloc.utils.miou_image on 5 cases (incl. empty unions)")
    for method in args.methods:
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == method][:args.smoke_n]
        ex = build_extractor(method, load_wm, task_root, args, device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        feats, meta = extract_all(ex, ds, idx, args.batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log(f"[{method}] features for {len(idx)} images: {time.time() - t0:.2f}s, peak GPU "
            f"{torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.2f} GB")
        per_img = 0
        for k, v in feats.items():
            log(f"   {k:34s} {tuple(v.shape)} {v.dtype} finite={bool(torch.isfinite(v.float()).all())}")
            per_img += v[0].numel() * v.element_size()
        log(f"   cached bytes/image {per_img / 1e6:.2f} MB -> 170 images {per_img * 170 / 1e9:.2f} GB")
        # 2) mask alignment: npz grid == official downsample of the 512 mask
        for m in meta:
            assert m["mask"].shape == (512, 512) and m["mask128"].shape == (GRID, GRID)
            assert np.array_equal(wutils.downsample_mask(m["mask"]), m["mask128"]), m["image_name"]
            assert set(np.unique(m["mask128"]).tolist()) <= {0, 1}
        log(f"[{method}] masks: shapes ok, binary, mask128 == downsample_mask(mask) for {len(meta)} images")
        if "agree" in feats:                                   # per-image signal check (diagnostic only)
            a = F.interpolate(feats["agree"].float(), size=GRID, mode="bilinear", align_corners=False)
            for ch, nm in enumerate(("logit", "sign")):
                d = []
                for i, m in enumerate(meta):
                    g = torch.from_numpy(m["mask128"]).bool()
                    if g.any() and (~g).any():
                        d.append(float(a[i, ch][g].mean() - a[i, ch][~g].mean()))
                log(f"   agreement-{nm} (inside - outside mask) per image: {np.round(d, 3).tolist()}")
        # 3) forward/backward and gradient checks
        maps = {n: feats[n].shape[1] for n in FEATURES[method]}
        head = SegHead(maps, feats["agree"].shape[1] if "agree" in feats else 0).to(device)
        log(f"[{method}] head params {sum(p.numel() for p in head.parameters()) / 1e6:.3f} M")
        tgt = torch.from_numpy(np.stack([m["mask128"] for m in meta])).float().to(device)
        head.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.startswith("cuda")):
            logits = head(to_device(feats, torch.arange(len(idx)), device))
        assert logits.shape == (len(idx), GRID, GRID), logits.shape
        loss = loss_fn(logits.float(), tgt, args.dice_weight)
        loss.backward()
        g = [p.grad for p in head.parameters()]
        assert all(x is not None for x in g) and sum(float(x.abs().sum()) for x in g) > 0
        for mod in ex.torch_modules:
            assert all((not p.requires_grad) and p.grad is None for p in mod.parameters())
        log(f"[{method}] logits {tuple(logits.shape)}, loss {loss.item():.4f}, head grads ok, "
            f"extractor frozen ({'torch modules' if ex.torch_modules else 'ONNX graph, no torch params'})")
        probs = torch.sigmoid(logits.float()).detach().cpu()
        log(f"[{method}] untrained head mIoU@0.5 {float(miou_batch(probs >= 0.5, tgt.cpu()).mean()):.4f} | "
            f"all-zero {float(miou_batch(torch.zeros_like(tgt.cpu()), tgt.cpu()).mean()):.4f} | "
            f"all-one {float(miou_batch(torch.ones_like(tgt.cpu()), tgt.cpu()).mean()):.4f}")
        sc = feats["scalars"]
        log(f"[{method}] global soft agreement {sc[:, 0].numpy().round(2)} bit acc {sc[:, 1].numpy().round(2)}")
        del ex
    log("SMOKE OK")


def train_parallel(args):
    """One child process per method, each pinned to its own GPU (CUDA_VISIBLE_DEVICES); the shared launcher
    stays a single srun task. Children write head_<method>.pt / history; the parent merges the summaries."""
    import subprocess
    ngpu = torch.cuda.device_count()
    procs = []
    for k, method in enumerate(args.methods):
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "train", "--child"] + \
              [a for a in sys.argv[2:] if a not in ("--parallel",)]
        if "--methods" in cmd:                                   # drop the parent's method list
            i = cmd.index("--methods"); j = i + 1
            while j < len(cmd) and not cmd[j].startswith("--"):
                j += 1
            del cmd[i:j]
        cmd += ["--methods", method]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % ngpu))
        logf = open(Path(args.out) / f"train_{method}.log", "w")
        log(f"launch [{method}] on GPU {k % ngpu}: log {logf.name}")
        procs.append((method, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
    codes = {}
    for method, p, logf in procs:
        codes[method] = p.wait(); logf.close()
        log(f"[{method}] exit code {codes[method]}")
        print(Path(logf.name).read_text()[-3000:], flush=True)
    summary = {}
    for method in args.methods:
        f = Path(args.out) / f"train_summary_{method}.json"
        if f.exists():
            summary.update(json.loads(f.read_text()))
    (Path(args.out) / "train_summary.json").write_text(json.dumps(summary, indent=1))
    log("TRAIN DONE (parallel)", json.dumps(summary))
    if any(codes.values()):
        raise SystemExit(f"child failures: {codes}")


def mode_train(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.parallel and not args.child and len(args.methods) > 1 and torch.cuda.device_count() > 1:
        return train_parallel(args)
    task_root, wutils, LDS, load_wm, device = common_setup(args)
    ds = LDS(task_root / args.data, "train")
    train_idx, val_idx = make_split(ds.meta, args.val_frac, args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    split = {"seed": args.seed, "val_frac": args.val_frac,
             "train": [ds.meta[i]["image_name"] for i in train_idx], "val": [ds.meta[i]["image_name"] for i in val_idx]}
    tmp = out_dir / f"split.json.{os.getpid()}"                 # deterministic content; atomic for parallel children
    tmp.write_text(json.dumps(split, indent=1)); os.replace(tmp, out_dir / "split.json")
    summary = {}
    for method in args.methods:
        set_seed(args.seed)
        tr = [i for i in train_idx if ds.meta[i]["method"] == method]
        va = [i for i in val_idx if ds.meta[i]["method"] == method]
        ex = build_extractor(method, load_wm, task_root, args, device)
        t0 = time.time()
        ftr, mtr = extract_all(ex, ds, tr, args.batch_size)
        fva, mva = extract_all(ex, ds, va, args.batch_size)
        log(f"[{method}] features train {len(tr)} / val {len(va)} in {time.time() - t0:.1f}s")
        model_info = {"method": method, "features": list(FEATURES[method]),
                      "extractor": type(ex.wm).__name__, "onnx_providers": getattr(ex, "providers", None)}
        del ex; torch.cuda.empty_cache()
        ytr = torch.from_numpy(np.stack([m["mask128"] for m in mtr])).float()
        yva = torch.from_numpy(np.stack([m["mask128"] for m in mva])).float()
        base = {"all_zero": float(miou_batch(torch.zeros_like(yva), yva).mean()),
                "all_one": float(miou_batch(torch.ones_like(yva), yva).mean())}
        maps = {n: ftr[n].shape[1] for n in FEATURES[method]}
        head = SegHead(maps, ftr["agree"].shape[1] if "agree" in ftr else 0).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = args.epochs * math.ceil(len(tr) / args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        best = {"miou": -1}
        history = []
        for ep in range(1, args.epochs + 1):
            head.train(); perm = torch.randperm(len(tr)); tl = 0
            for s in range(0, len(tr), args.batch_size):
                idx = perm[s:s + args.batch_size]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.startswith("cuda")):
                    logits = head(to_device(ftr, idx, device))
                loss = loss_fn(logits.float(), ytr[idx].to(device), args.dice_weight)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tl += loss.item() * len(idx)
            probs = predict_probs(head, fva, device, args.batch_size, args.amp)
            ths = score_thresholds(probs, yva)
            thr = max(ths, key=ths.get)
            history.append({"epoch": ep, "train_loss": tl / len(tr), "val_miou_at_0.5": ths[0.5],
                            "val_best_thr": thr, "val_best_miou": ths[thr]})
            if ths[thr] > best["miou"]:
                best = {"miou": ths[thr], "thr": thr, "epoch": ep, "miou_at_0.5": ths[0.5]}
                torch.save({"state_dict": head.state_dict(), "map_channels": maps,
                            "n_agree": ftr["agree"].shape[1] if "agree" in ftr else 0,
                            "threshold": thr, "config": vars(args), "split": split, "model": model_info,
                            "val_metrics": {"miou": ths[thr], "miou_at_0.5": ths[0.5], "epoch": ep,
                                            "threshold_grid": ths, **base}},
                           out_dir / f"head_{method}.pt")
            if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
                log(f"[{method}] ep {ep:3d} loss {tl / len(tr):.4f} val mIoU@0.5 {ths[0.5]:.4f} "
                    f"best-thr {thr} -> {ths[thr]:.4f} | best so far {best['miou']:.4f} (ep {best['epoch']})")
        summary[method] = {**best, **base, "n_train": len(tr), "n_val": len(va)}
        (out_dir / f"history_{method}.json").write_text(json.dumps(history, indent=1))
        (out_dir / f"train_summary_{method}.json").write_text(json.dumps({method: summary[method]}, indent=1))
        log(f"[{method}] BEST val mIoU {best['miou']:.4f} at thr {best['thr']} ep {best['epoch']} | "
            f"all-zero {base['all_zero']:.4f} all-one {base['all_one']:.4f}")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    log("TRAIN DONE", json.dumps(summary))


def load_head(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = SegHead(ck["map_channels"], ck["n_agree"]); head.load_state_dict(ck["state_dict"])
    return head.to(device).eval(), ck


def save_overlays(path, images, gts, preds, names, n=8):
    from PIL import Image, ImageDraw
    rows = []
    for img, g, p, nm in list(zip(images, gts, preds, names))[:n]:
        im = Image.fromarray(img).resize((256, 256))
        gi = Image.fromarray((np.asarray(g) * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST).convert("RGB")
        pi = Image.fromarray((np.asarray(p) * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST).convert("RGB")
        row = Image.new("RGB", (768, 272), "white")
        for j, x in enumerate((im, gi, pi)):
            row.paste(x, (256 * j, 16))
        ImageDraw.Draw(row).text((4, 2), f"{nm}  image | ground truth | prediction  "
                                          f"mIoU {float(miou_batch(torch.tensor(p)[None], torch.tensor(g)[None])[0]):.3f}",
                                 fill="black")
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (768, 272 * len(rows)), "white")
        for i, r in enumerate(rows):
            sheet.paste(r, (0, 272 * i))
        sheet.save(path)


def mode_evaluate(args):
    task_root, wutils, LDS, load_wm, device = common_setup(args)
    ds = LDS(task_root / args.data, "train")
    out_dir = Path(args.out); (out_dir / "viz").mkdir(parents=True, exist_ok=True)
    report, all_rows = {}, []
    for method in args.methods:
        head, ck = load_head(Path(args.ckpt_dir) / f"head_{method}.pt", device)
        val_names = set(ck["split"]["val"])
        va = [i for i, m in enumerate(ds.meta) if m["method"] == method and m["image_name"] in val_names]
        ex = build_extractor(method, load_wm, task_root, args, device)
        fva, mva = extract_all(ex, ds, va, args.batch_size); del ex
        probs = predict_probs(head, fva, device, args.batch_size, args.amp)
        yva = torch.from_numpy(np.stack([m["mask128"] for m in mva])).long()
        thr = ck["threshold"]; pred = (probs >= thr).long()
        mi = miou_batch(pred, yva)
        # cross-check with the official per-image function
        off = np.array([wutils.miou_image(p.numpy(), g.numpy()) for p, g in zip(pred, yva)])
        assert np.allclose(off, mi.numpy(), atol=1e-6)
        fw = np.array([bool(m["full_watermark"]) for m in mva]); cov = yva.float().flatten(1).mean(1).numpy()
        report[method] = {
            "threshold": thr, "n_val": len(va), "miou": float(mi.mean()),
            "all_zero": float(miou_batch(torch.zeros_like(yva), yva).mean()),
            "all_one": float(miou_batch(torch.ones_like(yva), yva).mean()),
            "miou_full_watermark_true": float(mi[torch.from_numpy(fw)].mean()) if fw.any() else None,
            "miou_full_watermark_false": float(mi[torch.from_numpy(~fw)].mean()) if (~fw).any() else None,
            "miou_coverage_lt_0.1": float(mi[torch.from_numpy(cov < 0.1)].mean()) if (cov < 0.1).any() else None,
            "miou_coverage_gt_0.9": float(mi[torch.from_numpy(cov > 0.9)].mean()) if (cov > 0.9).any() else None,
            "pixel_accuracy (not the metric)": float((pred == yva).float().mean()),
            "global_bit_acc_mean (not localisation)": float(fva["scalars"][:, 1].mean()),
        }
        all_rows += mi.tolist()
        order = np.argsort(mi.numpy())                                    # worst first, then best
        sel = list(order[:4]) + list(order[-4:])
        save_overlays(out_dir / "viz" / f"val_{method}.png", [ds[va[i]]["image"] for i in sel],
                      [yva[i].numpy() for i in sel], [pred[i].numpy() for i in sel], [mva[i]["image_name"] for i in sel])
        log(f"[{method}] val mIoU {report[method]['miou']:.4f} (thr {thr}) | all-zero {report[method]['all_zero']:.4f} "
            f"all-one {report[method]['all_one']:.4f}")
    if all_rows:
        report["overall"] = {"miou": float(np.mean(all_rows)), "n_val": len(all_rows)}
        log(f"OVERALL val mIoU {report['overall']['miou']:.4f} over {len(all_rows)} images")
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=1))
    log("EVAL DONE", json.dumps(report, indent=1))


def validate_submission(path, expected_names, max_mb):
    size = os.path.getsize(path) / 1e6
    sub = torch.load(path, weights_only=True)
    assert set(sub) == {"image_names", "masks"}, sub.keys()
    names, masks = sub["image_names"], sub["masks"]
    assert isinstance(names, list) and all(isinstance(n, str) for n in names)
    assert len(names) == len(set(names)), "duplicate names"
    assert set(names) == set(expected_names), "name set differs from test metadata"
    assert masks.shape == (len(expected_names), GRID, GRID), masks.shape
    assert masks.dtype in (torch.bool, torch.uint8), masks.dtype
    assert set(torch.unique(masks.to(torch.uint8)).tolist()) <= {0, 1}
    assert size <= max_mb, f"{size:.2f} MB > {max_mb} MB"
    return size


def mode_predict(args):
    task_root, wutils, LDS, load_wm, device = common_setup(args)
    ds = LDS(task_root / args.data, "test")
    names_all = [m["image_name"] for m in ds.meta]
    masks = torch.zeros(len(ds), GRID, GRID, dtype=torch.bool); done = np.zeros(len(ds), bool)
    for method in METHODS:
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == method]
        if not idx:
            continue
        head, ck = load_head(Path(args.ckpt_dir) / f"head_{method}.pt", device)
        ex = build_extractor(method, load_wm, task_root, args, device)
        for s in range(0, len(idx), 128):                                # stream: no full test cache
            chunk = idx[s:s + 128]
            f, meta = extract_all(ex, ds, chunk, args.batch_size)
            assert [m["image_name"] for m in meta] == [names_all[i] for i in chunk]
            probs = predict_probs(head, f, device, args.batch_size, args.amp)
            masks[chunk] = probs >= ck["threshold"]; done[chunk] = True
        del ex
        log(f"[{method}] predicted {len(idx)} test masks, thr {ck['threshold']}, "
            f"mean predicted coverage {masks[idx].float().mean():.3f}")
    assert done.all(), f"{(~done).sum()} test images not predicted"
    out = Path(args.submission); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names_all, "masks": masks}, out)
    size = validate_submission(out, names_all, args.max_mb)
    log(f"submission {out}: {len(names_all)} masks, {size:.2f} MB, reload + checks OK")


# --------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "train", "evaluate", "predict"])
    ap.add_argument("--task-root", default=".", help="folder with wmloc/, models/, release/")
    ap.add_argument("--data", default="release")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--work-models", default=f"work/{os.environ.get('USER', 'user')}/models",
                    help="where the StegaStamp feature ONNX copy goes (shared models/ stays untouched)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="disable bf16 autocast for the head")
    ap.add_argument("--out", default="output/baseline", help="train: checkpoints dir; evaluate: report dir")
    ap.add_argument("--ckpt-dir", default="output/baseline")
    ap.add_argument("--submission", default="output/baseline/submission.pt")
    ap.add_argument("--max-mb", type=float, default=20.0, help="PDF says 20 MB (README says 25): keep the stricter")
    ap.add_argument("--pylib", default=f"work/{os.environ.get('USER', 'user')}/pylib",
                    help="prepended to sys.path: private trustmark copy holding its downloaded weights "
                         "(compute nodes have no internet; the shared .venv has no TrustMark weights)")
    ap.add_argument("--shifts", type=int, default=4,
                    help="shift-and-stitch factor per axis for the 8x8 extractor maps (1 = off)")
    ap.add_argument("--parallel", action="store_true",
                    help="train: one process per method on separate GPUs (request --gres=gpu:4 in sbatch)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--smoke-n", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    log("args", json.dumps(vars(args)))
    {"smoke": mode_smoke, "train": mode_train, "evaluate": mode_evaluate, "predict": mode_predict}[args.mode](args)


if __name__ == "__main__":
    main()
