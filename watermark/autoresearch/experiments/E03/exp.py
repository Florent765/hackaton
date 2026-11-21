"""E03 experiments: per-bit evidence channels appended to the agreement stack (PixelSeal, TrustMark only).
E03_evidence      : + [std, q10, Z] computed with the TRUE message (per-image standardised)
E03_evidence_ctrl : + the same channels computed with a fixed RANDOM message (control: same capacity, no message)
StegaStamp identical to REF."""
from pathlib import Path

import torch

CACHE = Path("/p/project1/training2666/jacquet2/research_cache/E03")


def make_extra(key):
    def extra(c, split):
        if c["method"] not in ("pixelseal", "trustmark"):
            return {}
        o = torch.load(CACHE / f"{split}_{c['method']}_evidence.pt", weights_only=False)
        assert o["names"] == c["names"], "evidence not aligned"
        d = o[key].float()
        d = (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)
        return {"agree_extra": d}
    return extra


REGISTRY = {
    "E03_evidence": lambda cfg: (cfg(), make_extra("true"), None, "per-bit evidence std/q10/Z, true message"),
    "E03_evidence_ctrl": lambda cfg: (cfg(), make_extra("ctrl"), None, "same channels with a random message (control)"),
}
