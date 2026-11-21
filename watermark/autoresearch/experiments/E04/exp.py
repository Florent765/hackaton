"""E04 experiments: StegaStamp conv maps reprojected from the STN frame to the input-image frame.
E04_reproj        : dec_conv_3/5 maps replaced by their reprojection + a validity map (only change vs REF)
E04_reproj_novalid: same without the validity map (ablation)
Inverse warp: grid = affine_grid(inv(theta)) read in the STN-output frame (convention verified, corr 0.983).
PixelSeal / TrustMark untouched (identical to REF)."""
from pathlib import Path

import torch
import torch.nn.functional as F

CACHE = Path("/p/project1/training2666/jacquet2/research_cache/E04")
MAPS = ("gen_decoder/dec_conv_3/Relu:0", "gen_decoder/dec_conv_5/Relu:0")


def reproject(x, theta):
    """x [N,C,h,w] in STN-output frame -> input frame; returns (maps, valid [N,1,h,w])."""
    M = torch.cat([theta, torch.tensor([[[0., 0., 1.]]]).expand(len(theta), 1, 3)], 1)
    inv = torch.linalg.inv(M)[:, :2]
    out, val = [], []
    for s in range(0, len(x), 32):
        g = F.affine_grid(inv[s:s + 32], list(x[s:s + 32].shape), align_corners=False)
        out.append(F.grid_sample(x[s:s + 32], g, align_corners=False, padding_mode="zeros"))
        val.append(F.grid_sample(torch.ones_like(x[s:s + 32, :1]), g, align_corners=False))
    return torch.cat(out), torch.cat(val)


def make_extra(with_valid):
    def extra(c, split):
        if c["method"] != "stegastamp":
            return {}
        t = torch.load(CACHE / f"{split}_stegastamp_theta.pt", weights_only=False)
        assert t["names"] == c["names"], "theta not aligned"
        res = {}
        for n in MAPS:
            x = torch.log1p(c["feats"][n].float().clamp(min=0))            # same transform as REF (current)
            r, v = reproject(x, t["theta"])
            res[n] = r
        if with_valid:
            res["valid"] = v
        return res
    return extra


REGISTRY = {
    "E04_reproj": lambda cfg: (cfg(), make_extra(True), None, "StegaStamp maps reprojected to input frame + validity"),
    "E04_reproj_novalid": lambda cfg: (cfg(), make_extra(False), None, "StegaStamp maps reprojected, no validity map"),
}
