"""Autoresearch round on the 3 best leads (one change each; reference = N03_attr, same seeds).
Lead 1 (N03 StegaStamp attribution):
  AR_n03_rawstd : attribution standardised + fixed raw channel sign(x)*log1p(|x|)
  AR_n03_posneg : relu(attr) and relu(-attr), each per-image standardised
  AR_n03_dense  : attribution densified by input shifts (52x52, prepare_attr_dense.py) instead of 13x13
Lead 3 (TrustMark gradient x activation, N03b): on top of N03
  AR_tm_l3      : TrustMark layer3 attribution only
  AR_tm_l2      : TrustMark layer2 attribution only
(Lead 2 = N01 decoding variants: eval_n01_variants.py, no training.)"""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RC = Path("/p/project1/training2666/jacquet2/research_cache")
_s = importlib.util.spec_from_file_location("n03exp", HERE.parent / "N03" / "exp.py"); N03 = importlib.util.module_from_spec(_s); _s.loader.exec_module(N03)


def std(d):
    return (d - d.mean((2, 3), keepdim=True)) / d.std((2, 3), keepdim=True).clamp(min=1e-6)


def stega_variant(kind):
    def extra(c, split):
        if c["method"] != "stegastamp":
            return {}
        if kind == "dense":
            o = torch.load(RC / "AR" / f"{split}_stegastamp_attr_dense.pt", weights_only=False)
            assert o["names"] == c["names"]
            return {"agree_extra": std(o["attr"].float())}
        o = torch.load(RC / "N03" / f"{split}_stegastamp_attr.pt", weights_only=False)
        assert o["names"] == c["names"]
        a = o["attr"].float()
        if kind == "rawstd":
            return {"agree_extra": torch.cat([std(a), torch.sign(a) * torch.log1p(a.abs())], 1)}
        if kind == "posneg":
            return {"agree_extra": torch.cat([std(F.relu(a)), std(F.relu(-a))], 1)}
    return extra


def tm_layer(layer):
    stega = N03.make_extra("attr")

    def extra(c, split):
        if c["method"] == "stegastamp":
            return stega(c, split)
        if c["method"] != "trustmark":
            return {}
        o = torch.load(RC / "N03b" / f"{split}_trustmark_gxa.pt", weights_only=False)
        assert o["names"] == c["names"]
        return {"agree_extra": F.interpolate(std(o["true"][layer].float()), size=32, mode="bilinear", align_corners=False)}
    return extra


REGISTRY = {
    "AR_n03_rawstd": lambda cfg: (cfg(), stega_variant("rawstd"), None, "N03 attr standardised + raw signed-log channel"),
    "AR_n03_posneg": lambda cfg: (cfg(), stega_variant("posneg"), None, "N03 attr split into positive / negative parts"),
    "AR_n03_dense": lambda cfg: (cfg(), stega_variant("dense"), None, "N03 attr densified by input shifts (52x52)"),
    "AR_tm_l3": lambda cfg: (cfg(), tm_layer("layer3"), None, "N03 + TrustMark layer3 gxa only"),
    "AR_tm_l2": lambda cfg: (cfg(), tm_layer("layer2"), None, "N03 + TrustMark layer2 gxa only"),
}
