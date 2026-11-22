"""Shared helpers for branches B-JiT / C-consensus (diagnostics only; NOT the official score)."""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import ul_lib as L

SAVE = Path("/p/project1/training2666/jacquet2/ul/bjc")      # raw final state dicts (not exports)


def cutmix_u8(u8, seed=123):
    """Pseudo-control proxy: CutMix of random forget-image pairs (box 30-70% side), uint8, deterministic."""
    g = torch.Generator().manual_seed(seed); n = len(u8); perm = torch.randperm(n, generator=g)
    out = u8.clone()
    for i in range(n):
        s = int(224 * (0.3 + 0.4 * torch.rand(1, generator=g).item()))
        a, b = torch.randint(0, 224 - s + 1, (2,), generator=g).tolist()
        out[i, a:a + s, b:b + s] = u8[perm[i], a:a + s, b:b + s]
    return out


class Ctx:
    def __init__(self):
        t0 = time.time(); self.u8 = L.load_images(); self.tc = L.teacher_cache(self.u8)
        self.pl = self.tc["clean_0"].argmax(1)
        self.cm = cutmix_u8(self.u8); orig = L.load_original()
        self.cm_orig = L.logits(orig, self.cm, "clean"); del orig
        self.sd0 = L.original_state(); SAVE.mkdir(parents=True, exist_ok=True)
        print(f"ctx ready {time.time()-t0:.0f}s", flush=True)


def summarize(lgs, ctx, T=2.0):
    """lgs: dict kind -> logits (clean/mild/strong/diag on seed 0, 'cm' on cutmix proxy). Compact metrics."""
    out = {}
    for k, lg in lgs.items():
        ref = ctx.cm_orig if k == "cm" else ctx.tc[f"{k}_0"]
        p = F.log_softmax(lg / T, 1); q = F.log_softmax(ref / T, 1)
        kl = (q.exp() * (q - p)).sum(1)
        ce_v = F.cross_entropy(lg, ref.argmax(1), reduction="none")          # vs orig argmax on same view
        r = {"kl": round(float(kl.mean()), 4), "agree": round(float((lg.argmax(1) == ref.argmax(1)).float().mean()), 4),
             "ce": round(float(ce_v.mean()), 4), "ce_sd": round(float(ce_v.std()), 4),
             "conf": round(float(F.softmax(lg, 1).max(1).values.mean()), 4), "ncls": int(lg.argmax(1).unique().numel())}
        if k != "cm":
            ce_c = F.cross_entropy(lg, ctx.pl, reduction="none")              # vs CLEAN pseudo-label
            r["ce_clean_pl"] = round(float(ce_c.mean()), 4); r["ce_clean_pl_sd"] = round(float(ce_c.std()), 4)
            r["ce_q05_50_95"] = [round(float(v), 3) for v in torch.quantile(ce_c, torch.tensor([.05, .5, .95]))]
        out[k] = r
    return out


def evaluate(model, ctx, name, extra=None):
    lgs = {k: L.logits(model, ctx.u8, k, 0) for k in ("clean", "mild", "strong", "diag")}
    lgs["cm"] = L.logits(model, ctx.cm, "clean")
    d = summarize(lgs, ctx); d.update(extra or {})
    print("DIAG " + name + " " + json.dumps(d), flush=True); return d


def temp_control(ctx, target_clean_ce, name):
    """Calibration control = A1-style head scaling (scales weight+bias => logits*s exactly). Choose s matching the
    clean CE (vs pseudo-label) of a candidate; report the same metrics analytically from cached original logits."""
    lo, hi = 1e-3, 1.0
    f = lambda s: float(F.cross_entropy(ctx.tc["clean_0"] * s, ctx.pl))
    if f(hi) >= target_clean_ce:
        s = 1.0
    else:
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) < target_clean_ce else (mid, hi)
        s = (lo + hi) / 2
    lgs = {k: ctx.tc[f"{k}_0"] * s for k in ("clean", "mild", "strong", "diag")}; lgs["cm"] = ctx.cm_orig * s
    d = summarize(lgs, ctx); d["scale"] = round(s, 4)
    print("CTRL " + name + " " + json.dumps(d), flush=True); return d


def params_for(model, scope):
    for p in model.parameters():
        p.requires_grad_(False)
    nb = len(model.blocks)
    if scope == "all":
        ps = list(model.parameters())
    elif scope == "head":
        ps = list(model.head.parameters()) + list(model.norm.parameters())
    elif scope.startswith("last"):                                      # lastK: last K blocks + final norm + head
        k = int(scope[4:]); ps = [p for b in model.blocks[nb - k:] for p in b.parameters()]
        ps += list(model.norm.parameters()) + list(model.head.parameters())
    elif scope == "gamma":                                              # LayerScale gammas only
        ps = [p for n, p in model.named_parameters() if n.endswith(".gamma")]
    else:
        raise ValueError(scope)
    for p in ps:
        p.requires_grad_(True)
    return ps


def drift(model, sd0):
    sd = model.state_dict(); num = sum(float((sd[k].float().cpu() - sd0[k].float()).pow(2).sum()) for k in sd0)
    den = sum(float(sd0[k].float().pow(2).sum()) for k in sd0); return round((num / den) ** 0.5, 6)


def save(model, name):
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        print("skip save (non-finite)", flush=True); return
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, SAVE / f"{name}.pt")
