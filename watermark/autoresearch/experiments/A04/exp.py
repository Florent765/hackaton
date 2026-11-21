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


def make_extra(methods, key="soft", ctrl=False, base="n03"):
    base_fn = A01.make_extra([] if base == "n03" else ["a6"])

    def extra(c, split):
        out = base_fn(c, split)
        if c["method"] not in methods:
            return out
        o = torch.load(A04 / f"{split}_{c['method']}_crops.pt", weights_only=False)
        assert list(o["names"]) == list(c["names"]), "crop maps not aligned"
        x = o[key + ("_ctrl" if ctrl else "")].float() * 4
        out["agree_extra"] = torch.cat([out["agree_extra"], x], 1) if "agree_extra" in out else x
        return out
    return extra


ALL = ("pixelseal", "trustmark")          # StegaStamp: no crop signal in the probe (AUROC 0.56 true vs 0.60 ctrl)
REGISTRY = {
    "A04_crops": lambda cfg: (cfg(), make_extra(ALL), None, "N03 + crop-decoding soft-agreement maps (5 scales), PixelSeal+TrustMark"),
    "A04_crops_ctrl": lambda cfg: (cfg(), make_extra(ALL, ctrl=True), None, "same with a random message (control)"),
    "A04_crops_a6": lambda cfg: (cfg(), make_extra(ALL, base="a6"), None, "A01_a6 + crop-decoding maps, PixelSeal+TrustMark"),
}
