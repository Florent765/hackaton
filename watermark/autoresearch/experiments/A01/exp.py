"""A01 experiments: N03_attr + finer exact StegaStamp attribution maps (dec_conv_6/5/4) or shift-TTA of the N03 map.
Maps come from A01/prepare_deep.py (research_cache/A01). Each extra map is average-pooled to <= 32x32 and
standardised per image, then appended to the agreement stack after the N03 map. PixelSeal / TrustMark = REF."""
from pathlib import Path

import torch
import torch.nn.functional as F

N03 = Path("/p/project1/training2666/jacquet2/research_cache/N03")
A01 = Path("/p/project1/training2666/jacquet2/research_cache/A01")


def stdz(d):
    return (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)


def make_extra(keys, n03_key="attr", ctrl=False):
    def extra(c, split):
        if c["method"] != "stegastamp":
            return {}
        o = torch.load(N03 / f"{split}_stegastamp_attr.pt", weights_only=False)
        assert o["names"] == c["names"], "attr not aligned"
        chans = [F.interpolate(stdz(o[n03_key].float()), size=32, mode="bilinear", align_corners=False)] if n03_key else []
        if keys:
            d = torch.load(A01 / f"{split}_stegastamp_deep.pt", weights_only=False)
            assert d["names"] == c["names"], "deep attr not aligned"
            for k in keys:
                x = d[k + ("_ctrl" if ctrl else "")].float()
                x = F.adaptive_avg_pool2d(x, 32) if x.shape[-1] > 32 else F.interpolate(x, size=32, mode="bilinear", align_corners=False)
                chans.append(stdz(x))
        return {"agree_extra": torch.cat(chans, 1)}
    return extra


REGISTRY = {
    "A01_a6": lambda cfg: (cfg(), make_extra(["a6"]), None, "N03 + exact attribution at dec_conv_6 (25x25)"),
    "A01_a65": lambda cfg: (cfg(), make_extra(["a6", "a5"]), None, "N03 + attribution at dec_conv_6 and dec_conv_5 (pooled 32)"),
    "A01_a6_ctrl": lambda cfg: (cfg(), make_extra(["a6"], ctrl=True), None, "N03 + dec_conv_6 attribution with random message (control)"),
    "A01_tta": lambda cfg: (cfg(), make_extra(["tta7"]), None, "N03 + shift-TTA (4x4, 8px) of the conv7 attribution"),
}
