"""A10 experiments: null-calibrated window statistics and decoding-refined rectangles (A10/prepare_win.py) appended
to the A04_rect_a65 agreement stack (PixelSeal, TrustMark; StegaStamp unchanged = A01_a65 maps).
Channels per method (32x32 cells of 16 px, from z = S1/sqrt(S2) of every window, see prepare_win.py):
  zmax   : max z over the windows covering the cell (all 17 shapes pooled; z is comparable across shapes)      /3
  con    : max z over covering windows - max z over windows NOT covering the cell (complement verification)     /3
  post_b : window posterior p(w) = softmax_w(b z_w), map = sum_w p(w) * area fraction of the cell in w, b in 0.5, 1  x4
  rbest  : area fraction of the best refined box x sigmoid(z_best - 5)                                             x4
  rmix   : sum_i softmax(rz)_i area fraction of refined box i, x sigmoid(max rz - 5)                               x4"""
import importlib.util
from pathlib import Path

import torch

A10 = Path("/p/project1/training2666/jacquet2/research_cache/A10")
HERE = Path(__file__).resolve().parent


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), HERE.parent / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


A04 = _load("A04/exp.py")


def load_win(split, method):
    parts = sorted(A10.glob(f"{split}_{method}_win_*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
    assert parts, f"no A10 window cache for {split} {method}"
    ps = [torch.load(q, weights_only=False) for q in parts]
    o = {k: torch.cat([q[k] for q in ps]) for k in ps[0] if isinstance(ps[0][k], torch.Tensor) and k not in ("box", "shape_idx")}
    o["names"] = [n for q in ps for n in q["names"]]; o["box"] = ps[0]["box"]
    return o


def frac_cov(tlbr):
    """tlbr [B,4] (px, 512 frame) -> [B,32,32] fraction of each 16-px cell inside the box"""
    tlbr = tlbr.float(); e = torch.arange(32, dtype=torch.float32, device=tlbr.device) * 16
    oy = (torch.minimum(tlbr[:, 2:3], e + 16) - torch.maximum(tlbr[:, 0:1], e)).clamp(min=0) / 16
    ox = (torch.minimum(tlbr[:, 3:4], e + 16) - torch.maximum(tlbr[:, 1:2], e)).clamp(min=0) / 16
    return oy[:, :, None] * ox[:, None, :]


def win_maps(o, suf, chans, dev="cuda"):
    z = (o["s1" + suf].float() / o["s2" + suf].float().clamp(min=1e-6).sqrt()).to(dev)            # [N, W]
    t, l, h, w = o["box"].float().unbind(1)
    fc = frac_cov(torch.stack([t, l, t + h, l + w], 1).to(dev))                                    # [W,32,32]
    cov = fc > 0
    out = []
    for i in range(0, len(z), 8):
        zi = z[i:i + 8]; ch = []
        zin = torch.where(cov[None], zi[:, :, None, None], torch.tensor(-1e4, device=dev)).amax(1)
        if "zmax" in chans:
            ch.append(zin / 3)
        if "con" in chans:
            zout = torch.where(cov[None], torch.tensor(-1e4, device=dev), zi[:, :, None, None]).amax(1)
            ch.append(((zin - zout) / 3).clamp(-4, 4))
        for b in (0.5, 1.0):
            if f"post{b}" in chans:
                ch.append(torch.einsum("nw,wyx->nyx", torch.softmax(b * zi, 1), fc) * 4)
        if "rbest" in chans or "rmix" in chans:
            rz = o["rz" + suf][i:i + 8].float().to(dev); rb = o["rbox" + suf][i:i + 8].to(dev)       # [n,4], [n,4,4]
            rc = frac_cov(rb.reshape(-1, 4)).view(len(rz), rz.shape[1], 32, 32)
            mx, am = rz.max(1)
            gate = torch.sigmoid(mx - 5)[:, None, None]
            if "rbest" in chans:
                ch.append(rc[torch.arange(len(rz), device=dev), am] * gate * 4)
            if "rmix" in chans:
                ch.append(torch.einsum("ni,niyx->nyx", torch.softmax(rz, 1), rc) * gate * 4)
        out.append(torch.stack(ch, 1))
    return torch.cat(out).cpu()


GRID = ("zmax", "con", "post0.5", "post1.0")
REF = ("rbest", "rmix")


def make_extra(chans, methods=("pixelseal", "trustmark"), ctrl=False):
    base_fn = A04.make_extra(A04.ALL, base="a65", rect=True)      # = A04_rect_a65 inputs

    def extra(c, split):
        out = base_fn(c, split)
        if c["method"] not in methods:
            return out
        o = load_win(split, c["method"])
        assert list(o["names"]) == list(c["names"]), "A10 window maps not aligned"
        x = win_maps(o, "_ctrl" if ctrl else "", chans)
        out["agree_extra"] = torch.cat([out["agree_extra"], x], 1)
        return out
    return extra


REGISTRY = {
    "A10_win": lambda cfg: (cfg(), make_extra(GRID), None, "A04_rect_a65 + null-calibrated window maps (zmax, contrast, posteriors)"),
    "A10_ref": lambda cfg: (cfg(), make_extra(GRID + REF), None, "A10_win + decoding-refined rectangle maps"),
    "A10_refonly": lambda cfg: (cfg(), make_extra(REF), None, "A04_rect_a65 + decoding-refined rectangle maps only"),
    "A10_ref_ctrl": lambda cfg: (cfg(), make_extra(GRID + REF, ctrl=True), None, "A10_ref with random-message window stats (control)"),
}
