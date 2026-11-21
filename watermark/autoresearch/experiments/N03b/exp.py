"""N03b experiments: gradient x activation message attribution at intermediate layers for PixelSeal / TrustMark,
appended to their agreement stack, ON TOP OF N03 (N03 only changes StegaStamp, N03b only PixelSeal/TrustMark, so
per-method effects are separable). Reference for comparison: N03_attr (same seeds).
N03b_true : true-message attribution maps (per-image standardised), all available layers
N03b_ctrl : same with a random message (control)"""
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CACHE = Path("/p/project1/training2666/jacquet2/research_cache/N03b")
_spec = importlib.util.spec_from_file_location("n03exp", HERE.parent / "N03" / "exp.py")
N03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(N03)


def make_extra(key):
    stega = N03.make_extra("attr")

    def extra(c, split):
        if c["method"] == "stegastamp":
            return stega(c, split)
        o = torch.load(CACHE / f"{split}_{c['method']}_gxa.pt", weights_only=False)
        assert o["names"] == c["names"], "gxa not aligned"
        maps = []
        for v in o[key].values():
            d = v.float()
            d = (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)
            maps.append(torch.nn.functional.interpolate(d, size=32, mode="bilinear", align_corners=False))
        return {"agree_extra": torch.cat(maps, 1)}
    return extra


REGISTRY = {
    "N03b_true": lambda cfg: (cfg(), make_extra("true"), None, "N03 + PS/TM gradient x activation attribution (true msg)"),
    "N03b_ctrl": lambda cfg: (cfg(), make_extra("ctrl"), None, "N03 + PS/TM gxa attribution with random message"),
}
