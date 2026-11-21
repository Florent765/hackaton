"""E05 experiments, ON TOP OF N03 (current confirmed best; residual branch is a new modality for all methods).
E05_res      : + 10 residual channels (128 grid) as an extra projected map
E05_grad     : + derivative channels only (dx, dy mean/rms) - control "RGB gradients only"
E05_shuffled : + residual channels permuted across images of the same method (fixed seed) - diagnostic control
Reference for comparison: N03_attr (same seeds)."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = Path("/p/project1/training2666/jacquet2/research_cache/E05")
_spec = importlib.util.spec_from_file_location("n03exp", HERE.parent / "N03" / "exp.py")
N03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(N03)


def make_extra(mode):
    stega = N03.make_extra("attr")

    def extra(c, split):
        out = stega(c, split) if c["method"] == "stegastamp" else {}
        o = torch.load(CACHE / f"{split}_{c['method']}_res.pt", weights_only=False)
        assert o["names"] == c["names"], "residuals not aligned"
        r = o["res"].float()
        if mode == "grad":
            r = r[:, 6:10]
        elif mode == "shuffled":
            r = r[torch.from_numpy(np.random.RandomState(5).permutation(len(r)))]
        out["residuals"] = r
        return out
    return extra


REGISTRY = {
    "E05_res": lambda cfg: (cfg(), make_extra("all"), None, "N03 + native multiband residuals (10 ch, 128 grid)"),
    "E05_grad": lambda cfg: (cfg(), make_extra("grad"), None, "N03 + derivative channels only (control)"),
    "E05_shuffled": lambda cfg: (cfg(), make_extra("shuffled"), None, "N03 + residuals shuffled across images (control)"),
}
