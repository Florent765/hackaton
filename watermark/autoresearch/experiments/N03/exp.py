"""N03 experiments: StegaStamp exact attribution map appended to its agreement stack (STN-output frame, kept
unwarped since E04 showed reprojection hurts).
N03_attr      : + attribution with the TRUE message (per-image standardised)
N03_attr_ctrl : + attribution with a RANDOM message (control, same capacity)
PixelSeal / TrustMark identical to REF."""
from pathlib import Path

import torch

CACHE = Path("/p/project1/training2666/jacquet2/research_cache/N03")


def make_extra(key):
    def extra(c, split):
        if c["method"] != "stegastamp":
            return {}
        o = torch.load(CACHE / f"{split}_stegastamp_attr.pt", weights_only=False)
        assert o["names"] == c["names"], "attr not aligned"
        d = o[key].float()
        d = (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)
        return {"agree_extra": d}
    return extra


REGISTRY = {
    "N03_attr": lambda cfg: (cfg(), make_extra("attr"), None, "StegaStamp exact flatten attribution, true message"),
    "N03_attr_ctrl": lambda cfg: (cfg(), make_extra("ctrl"), None, "same with random message (control)"),
}
