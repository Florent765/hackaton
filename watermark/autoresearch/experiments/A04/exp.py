"""A04 experiments: crop-decoding verifier maps (A04/prepare_crops.py) appended to the agreement stack.
Per method, 5 channels (one per window size): max over covering crops of the soft agreement with the message
(x4, no per-image standardisation: the absolute level is meaningful). Built on N03_attr (StegaStamp N03 map kept)
or on A01_a6 (N03 + dec_conv_6 map)."""
import importlib.util
from pathlib import Path

import torch

A04 = Path("/p/project1/training2666/jacquet2/research_cache/A04")
HERE = Path(__file__).resolve().parent


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), HERE.parent / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


A01 = _load("A01/exp.py")


def load_crops(split, method):
    """single file, or chunks {split}_{method}_crops_{start}.pt written by prepare_crops.py <m> <split> a b"""
    p = A04 / f"{split}_{method}_crops.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    parts = sorted(A04.glob(f"{split}_{method}_crops_*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
    ps = [torch.load(q, weights_only=False) for q in parts]
    o = {k: torch.cat([q[k] for q in ps]) for k in ("soft", "ba", "soft_ctrl", "ba_ctrl")}
    o["names"] = [n for q in ps for n in q["names"]]
    return o


def load_rect(split, method):
    parts = sorted(A04.glob(f"{split}_{method}_rect_*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
    ps = [torch.load(q, weights_only=False) for q in parts]
    o = {k: torch.cat([q[k] for q in ps]) for k in ("soft", "soft_ctrl")}
    o["names"] = [n for q in ps for n in q["names"]]
    return o


def make_extra(methods, key="soft", ctrl=False, base="n03", rect=False):
    base_fn = A01.make_extra({"n03": [], "a6": ["a6"], "a65": ["a6", "a5"]}[base])

    def extra(c, split):
        out = base_fn(c, split)
        if c["method"] not in methods:
            return out
        o = load_crops(split, c["method"])
        assert list(o["names"]) == list(c["names"]), "crop maps not aligned"
        x = o[key + ("_ctrl" if ctrl else "")].float() * 4
        if rect:                                   # + non-square windows (prepare_crops_rect.py), 12 channels
            r = load_rect(split, c["method"])
            assert list(r["names"]) == list(c["names"]), "rect maps not aligned"
            x = torch.cat([x, r[key + ("_ctrl" if ctrl else "")].float() * 4], 1)
        out["agree_extra"] = torch.cat([out["agree_extra"], x], 1) if "agree_extra" in out else x
        return out
    return extra


ALL = ("pixelseal", "trustmark")          # StegaStamp: no crop signal in the probe (AUROC 0.56 true vs 0.60 ctrl)
REGISTRY = {
    "A04_crops": lambda cfg: (cfg(), make_extra(ALL), None, "N03 + crop-decoding soft-agreement maps (5 scales), PixelSeal+TrustMark"),
    "A04_crops_ctrl": lambda cfg: (cfg(), make_extra(ALL, ctrl=True), None, "same with a random message (control)"),
    "A04_crops_a6": lambda cfg: (cfg(), make_extra(ALL, base="a6"), None, "A01_a6 + crop-decoding maps, PixelSeal+TrustMark"),
    "A04_rect_a65": lambda cfg: (cfg(), make_extra(ALL, base="a65", rect=True), None, "A04_crops_a65 + non-square crop windows (PS, TM)"),
    "A04_crops_a65": lambda cfg: (cfg(), make_extra(ALL, base="a65"), None, "A01_a65 (SS) + crop-decoding maps (PS, TM)"),
}
