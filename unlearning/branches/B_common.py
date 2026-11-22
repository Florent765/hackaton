"""Shared helpers for branches B3/B4/B5 (diagnostics only; pseudo-labels = original argmax)."""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import json
import time
import torch
import torch.nn.functional as F
import ul_lib as L

SAVE = L.CACHE.parent / "b345"            # project1 (GPU node only); state dicts kept here, exported on demand


def setup():
    t0 = time.time(); u8 = L.load_images(); tc = L.teacher_cache(u8); sd0 = L.original_state()
    g = torch.Generator().manual_seed(1234); perm = torch.randperm(len(u8), generator=g)
    fit, held = perm[:2500].sort().values, perm[2500:].sort().values
    print(f"setup {time.time()-t0:.0f}s N={len(u8)}", flush=True)
    return u8, tc, sd0, fit, held


def model_from(sd):
    m = L.build_model(); m.load_state_dict(sd, strict=True); return m.to(L.DEV).eval()


def short(d):
    """compact per-view summary of an L.full_diag dict"""
    return {k: {"kl": v["kl_orig_sub_T2"], "ce": v["ce_pseudo_mean"], "ce_sd": v["ce_pseudo_std"], "ceq": v["ce_q05_50_95"],
                "conf": v["conf_mean"], "chg": v["argmax_change"]} for k, v in d.items()}


def eval_grid(sd0, sd1, name, u8, tc, alphas=(0.25, 0.5, 1.0)):
    out = {}
    for a in alphas:
        sd = L.interpolate(sd0, sd1, a); m = model_from(sd); d = L.full_diag(m, u8, tc)
        L.report(f"{name}_a{a}", d); print("SHORT", name, a, json.dumps(short(d)), flush=True); out[a] = d
        del m; torch.cuda.empty_cache()
    return out


def save_sd(sd, name):
    SAVE.mkdir(parents=True, exist_ok=True); p = SAVE / f"{name}.pt"
    torch.save({k: v.detach().cpu().float() for k, v in sd.items()}, p); print("SAVED", p, flush=True); return p


def load_sd(name):
    return torch.load(SAVE / f"{name}.pt", map_location="cpu", weights_only=True)


def kl_t(lg_t, lg_s, T=2.0):
    """KL(teacher || student) at temperature T, per-sample mean, times T^2"""
    q = F.log_softmax(lg_t / T, 1); p = F.log_softmax(lg_s / T, 1)
    return (q.exp() * (q - p)).sum(1).mean() * T * T
