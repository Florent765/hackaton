"""autoresearch harness for watermark localisation -- FIXED, the agent does not modify this file.

  python prepare.py            build the frozen-feature caches (train + test, all methods) once, on a GPU node
  from prepare import ...      fixed API used by train.py:
      METHODS, K, TIME_BUDGET, load_cache(method, split), evaluate_oof(probs_or_masks), write_submission(...)

Features come from baseline.py's FrozenExtractor (organiser wrappers, frozen, eval mode, shift-and-stitch 4,
StegaStamp ONNX copy with decoder conv outputs). Metric: official two-class per-image mIoU (wmloc.utils), computed
on out-of-fold predictions of a K-fold split stratified by method x full_watermark (fixed seed). Source-image
grouping is impossible (hash names, distinct messages)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK_ROOT = Path(os.environ.get("AR_TASK_ROOT", "/p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task"))
USER = os.environ.get("USER", "user")
CACHE = Path(os.environ.get("AR_CACHE", TASK_ROOT / "work" / USER / "ar_cache"))
METHODS = ("pixelseal", "trustmark", "stegastamp")
K = 3
SEED = 0
TIME_BUDGET = 300          # seconds of training+eval for one experiment (all methods, all folds)
GRID = 128

sys.path.insert(0, str(HERE.parent))                     # baseline.py lives one level up
sys.path.insert(0, str(TASK_ROOT))
sys.path.insert(0, str(TASK_ROOT / "work" / USER / "pylib"))


def _folds(meta):
    rng = np.random.RandomState(SEED); fold = {}
    strata = {}
    for m in meta:
        strata.setdefault((m["method"], bool(m.get("full_watermark"))), []).append(m["image_name"])
    for key in sorted(strata):
        names = sorted(strata[key]); rng.shuffle(names)
        for i, n in enumerate(names):
            fold[n] = i % K
    return fold


def build():
    import baseline as B
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    for split in ("train", "test"):
        ds = LocalisationDataset(TASK_ROOT / "release", split)
        fold = _folds(ds.meta) if split == "train" else {}
        for method in METHODS:
            out = CACHE / f"{split}_{method}.pt"
            if out.exists():
                print("exists", out, flush=True); continue
            t0 = time.time()
            ex = B.FrozenExtractor(method, load_watermark, TASK_ROOT / "models", TASK_ROOT / "work" / USER / "models",
                                   dev, shifts=4)
            idx = [i for i, m in enumerate(ds.meta) if m["method"] == method]
            feats, meta = B.extract_all(ex, ds, idx, 16)
            rec = {"feats": feats, "names": [m["image_name"] for m in meta], "method": method,
                   "feature_names": list(B.FEATURES[method]), "shifts": 4}
            if split == "train":
                rec["masks"] = torch.from_numpy(np.stack([m["mask128"] for m in meta])).to(torch.uint8)
                rec["full_watermark"] = torch.tensor([bool(m["full_watermark"]) for m in meta])
                rec["fold"] = torch.tensor([fold[m["image_name"]] for m in meta])
            tmp = out.with_suffix(".tmp"); torch.save(rec, tmp); os.replace(tmp, out)
            del ex; torch.cuda.empty_cache()
            print(f"{split} {method}: {len(idx)} images, {time.time() - t0:.0f}s, "
                  f"{out.stat().st_size / 1e9:.2f} GB -> {out}", flush=True)
    print("PREPARE DONE", flush=True)


def load_cache(method: str, split: str = "train") -> dict:
    """dict(feats={name: tensor [N,...]}, names, masks [N,128,128] uint8 (train), full_watermark, fold (train))."""
    return torch.load(CACHE / f"{split}_{method}.pt", map_location="cpu", weights_only=False)


def _miou(pred, gt):
    from wmloc.utils import miou_image
    return miou_image(pred, gt)


def evaluate_oof(preds: dict) -> dict:
    """preds: {method: uint8/bool tensor [N,128,128]} of OUT-OF-FOLD binary masks, in load_cache order.
    Returns official mIoU overall (all train images equally weighted) and per method. Ground truth."""
    res, allv = {}, []
    for method in METHODS:
        gt = load_cache(method)["masks"].numpy()
        p = preds[method].to(torch.uint8).numpy()
        assert p.shape == gt.shape, (method, p.shape, gt.shape)
        assert set(np.unique(p).tolist()) <= {0, 1}
        v = [_miou(a, b) for a, b in zip(p, gt)]
        res[method] = float(np.mean(v)); allv += v
    res["val_miou"] = float(np.mean(allv))
    return res


def write_submission(names: list, masks: torch.Tensor, path, max_mb: float = 20.0) -> float:
    """Save and validate a submission against the test metadata (keys, names, dupes, shape, binary, size)."""
    expected = [m["image_name"] for m in json.loads((TASK_ROOT / "release/test/metadata.json").read_text())]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    order = {n: i for i, n in enumerate(names)}
    assert len(order) == len(names), "duplicate names"
    masks = masks.to(torch.bool)
    full = torch.stack([masks[order[n]] for n in expected])
    torch.save({"image_names": expected, "masks": full}, path)
    sub = torch.load(path, weights_only=True)
    assert set(sub) == {"image_names", "masks"} and sub["image_names"] == expected
    assert sub["masks"].shape == (len(expected), GRID, GRID) and sub["masks"].dtype == torch.bool
    size = path.stat().st_size / 1e6
    assert size <= max_mb, size
    return size


if __name__ == "__main__":
    build()
