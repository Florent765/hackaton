"""B3 - Bayesian division, minimal Laplace/influence version on a cheap subset P = last block (blocks.11) + head.
p(th|D_R) ~ p(th|D_tot)/p(D_F|th)  ->  Newton removal step  th_R = th0 + H_tot^{-1} grad_P NLL_F(th0).
- forget likelihood: clean forget images with HARD pseudo-labels (original argmax). The 'soft own prediction' version
  is identically zero in expectation (sum_y p(y) grad log p(y) = grad sum_y p(y) = 0): we compute and print it as a check.
- H_tot proxy: true Fisher (labels sampled from the model) on mild+strong views of the forget images.
  (a) diagonal, (b) K-FAC per linear layer (diag for non-linear params), (c) SWAG-like population (snapshots of SGD
  with sampled-label NLL on the views, low-rank covariance).  First test = cosines between the update directions.
Then each direction is scaled to a fixed KL(orig||sub) on held-out mild views and interpolated (alpha grid); random
direction with the same norm is the control.   usage: python B3_bayes.py [target_kl]"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import json
import math
import time
import torch
import torch.nn.functional as F
import ul_lib as L
import B_common as C

TARGETS = [float(t) for t in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.05]
DIRS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["diag", "kfac", "swag", "random", "grad"]
NTOT = 50000.0            # assumed size of the original training set (only sets the 'principled' Newton scale)
t00 = time.time()
u8, tc, sd0, fit, held = C.setup()
m = C.model_from(sd0)
for p in m.parameters(): p.requires_grad_(False)
tail_mods = {"qkv": m.blocks[11].attn.qkv, "proj": m.blocks[11].attn.proj, "fc1": m.blocks[11].mlp.fc1,
             "fc2": m.blocks[11].mlp.fc2, "head": m.head}
P = {n: p for n, p in m.named_parameters() if n.startswith("blocks.11.") or n.startswith("head.") or n.startswith("norm.")}
for p in P.values(): p.requires_grad_(True)
names = list(P)


@torch.no_grad()
def prefix(x):
    with torch.autocast("cuda", dtype=torch.float16):
        h = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
        for b in m.blocks[:11]:
            h = b(h)
    return h.half()


def tail(h):
    return m.forward_head(m.norm(m.blocks[11](h.float())))


def cache(idx, kind, seed):
    return torch.cat([prefix(L.view(L.prep(u8[idx[s:s + 128]]), kind, seed + s)) for s in range(0, len(idx), 128)])


# sanity: prefix+tail == model
with torch.no_grad():
    x = L.prep(u8[:8]); ref = m(x).float(); chk = tail(prefix(x)).float()
print("prefix/tail max abs diff", float((ref - chk).abs().max()), flush=True)

# ---------------- forget gradient (clean, hard pseudo-labels) + soft-own-target check
pl = tc["clean_0"].argmax(1)
g = {n: torch.zeros_like(p) for n, p in P.items()}; gs = {n: torch.zeros_like(p) for n, p in P.items()}
for s in range(0, len(u8), 128):
    b = list(range(s, min(s + 128, len(u8)))); h = prefix(L.prep(u8[b])); lg = tail(h)
    loss = F.cross_entropy(lg, pl[b].to(L.DEV), reduction="sum")
    gr = torch.autograd.grad(loss, list(P.values()), retain_graph=True)
    for n, t in zip(names, gr): g[n] += t
    ps = F.softmax(lg, 1).detach(); loss2 = -(ps * F.log_softmax(lg, 1)).sum()
    gr2 = torch.autograd.grad(loss2, list(P.values()))
    for n, t in zip(names, gr2): gs[n] += t
flat = lambda d: torch.cat([d[n].flatten() for n in names])
G = flat(g); GS = flat(gs)
print(f"forget grad: |g_hard|={G.norm():.4e}  |g_soft_own_target|={GS.norm():.4e} (expected ~0: sum_y p grad log p = 0)", flush=True)
print("per-param |g|:", json.dumps({n: float(g[n].norm()) for n in names if g[n].norm() > 0.05 * G.norm()}), flush=True)

# ---------------- 'total data' proxy: views of the FIT images (mild + strong), true Fisher with sampled labels
NV = 512
vidx = fit[torch.randperm(len(fit), generator=torch.Generator().manual_seed(5))[:NV]].tolist()
HV = torch.cat([cache(vidx[: NV // 2], "mild", 777), cache(vidx[NV // 2:], "strong", 777)])
fdiag = {n: torch.zeros_like(p) for n, p in P.items()}
A = {k: 0 for k in tail_mods}; Gk = {k: 0 for k in tail_mods}; acts = {}; gouts = {}
hooks = []
for k, mod in tail_mods.items():
    def fh(md, i, o, k=k):
        acts[k] = i[0].detach()
        if o.requires_grad:
            o.register_hook(lambda gr, k=k: gouts.__setitem__(k, gr.detach()))
    hooks.append(mod.register_forward_hook(fh))
gen = torch.Generator(device=L.DEV).manual_seed(0)
t0 = time.time()
for i in range(len(HV)):
    lg = tail(HV[i:i + 1]); y = torch.multinomial(F.softmax(lg.detach(), 1), 1, generator=gen)[:, 0]
    loss = F.cross_entropy(lg, y)
    gr = torch.autograd.grad(loss, list(P.values()))
    for n, t in zip(names, gr): fdiag[n] += t * t / len(HV)
    for k in tail_mods:
        a = acts[k].reshape(-1, acts[k].shape[-1]); d = gouts[k].reshape(-1, gouts[k].shape[-1])
        A[k] = A[k] + a.T @ a / a.shape[0] / len(HV); Gk[k] = Gk[k] + d.T @ d / len(HV)
for hk in hooks: hk.remove()
print(f"fisher diag+kfac {time.time()-t0:.0f}s", flush=True)

# (a) diagonal
lam_d = 1e-3 * float(flat(fdiag).mean())
Da = {n: g[n] / (fdiag[n] + lam_d) for n in names}
# (b) K-FAC for linear weights, diagonal for everything else
Db = {}
for n in names:
    k = next((k for k in tail_mods if n in (f"blocks.11.attn.{k}.weight", f"blocks.11.mlp.{k}.weight", f"{k}.weight")), None)
    if k is None:
        Db[n] = Da[n]; continue
    Ak, Gm = A[k], Gk[k]
    da = 1e-3 * float(Ak.diagonal().mean()); dg = 1e-3 * float(Gm.diagonal().mean())
    Ainv = torch.linalg.inv(Ak + da * torch.eye(len(Ak), device=L.DEV)); Ginv = torch.linalg.inv(Gm + dg * torch.eye(len(Gm), device=L.DEV))
    Db[n] = Ginv @ g[n] @ Ainv
# (c) SWAG-like population: SGD with sampled-label NLL on the view features, snapshots
sd_tail0 = {n: P[n].detach().clone() for n in names}
opt = torch.optim.SGD(list(P.values()), lr=1e-3, momentum=0.9)
snaps = []; t0 = time.time()
for step in range(220):
    b = torch.randint(0, len(HV), (32,)); lg = tail(HV[b])
    y = torch.multinomial(F.softmax(lg.detach(), 1), 1, generator=gen)[:, 0]
    loss = F.cross_entropy(lg, y); opt.zero_grad(); loss.backward(); opt.step()
    if step >= 40 and step % 20 == 0:
        snaps.append(flat({n: P[n].detach() for n in names}).clone())
with torch.no_grad():
    for n in names: P[n].copy_(sd_tail0[n])
th0 = flat(sd_tail0); Dev = torch.stack(snaps) - torch.stack(snaps).mean(0)          # K x D
print(f"swag {len(snaps)} snapshots {time.time()-t0:.0f}s, mean |dev|={Dev.norm(dim=1).mean():.3e}", flush=True)
Dc = (Dev.T @ (Dev @ G)) / max(1, len(snaps) - 1)

DA, DB = flat(Da), flat(Db)
dirs = {"grad": G, "diag": DA, "kfac": DB, "swag": Dc}
torch.manual_seed(0); dirs["random"] = torch.randn_like(G)
cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-30))
print("COSINES", json.dumps({f"{a}|{b}": round(cos(dirs[a], dirs[b]), 4) for i, a in enumerate(dirs) for b in list(dirs)[i + 1:]}), flush=True)
print("names order", names[-3:], flush=True)
print("COSINES_headonly", json.dumps({f"{a}|{b}": round(cos(dirs[a][-100 * 768 - 100:], dirs[b][-100 * 768 - 100:]), 4)
                                      for i, a in enumerate(dirs) for b in list(dirs)[i + 1:]}), flush=True)
# principled Newton scale for the diagonal: (NTOT*F + prior)^-1 g with prior precision 1
newton = G / (NTOT * flat(fdiag) + 1.0)
print(f"principled diag Newton step |d|={newton.norm():.3e} vs |theta_P|={th0.norm():.3e}", flush=True)


def sd_with(delta):
    sd = dict(sd0); off = 0
    for n in names:
        k = P[n].numel(); sd[n] = (sd0[n].float() + delta[off:off + k].view_as(P[n]).cpu()); off += k
    return sd


HIDX = held[:400].tolist()
REF_MILD = L.logits(C.model_from(sd0), u8, "mild", 0, idx=HIDX)      # same idx-call => identical views (seed+offset)


@torch.no_grad()
def kl_mild(delta):
    mm = C.model_from(sd_with(delta))
    lg = L.logits(mm, u8, "mild", 0, idx=HIDX); r = L.diag(lg, REF_MILD)["kl_orig_sub_T2"]; del mm; return r


print("KL_mild check at delta=0 (must be ~0):", kl_mild(torch.zeros_like(G)), flush=True)


print("KL_mild at principled Newton scale:", kl_mild(newton), flush=True)
results = {}
for TARGET, name in [(t, n) for t in TARGETS for n in DIRS]:
    d = dirs[name] / dirs[name].norm()
    lo, hi = 0.0, 1.0
    while kl_mild(hi * d) < TARGET and hi < 1e4: lo, hi = hi, hi * 4
    for _ in range(7):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kl_mild(mid * d) < TARGET else (lo, mid)
    s = hi; print(f"{name}: scale for KL_mild={TARGET}: |delta|={s:.3e} ({s/float(th0.norm()):.4f} rel)", flush=True)
    sd1 = sd_with(s * d)
    if name in ("diag", "kfac", "random"):
        res = C.eval_grid(sd0, sd1, f"B3_{name}_kl{TARGET}", u8, tc)
        if name != "random": C.save_sd(sd1, f"B3_{name}_kl{TARGET}")
    else:
        d1 = L.full_diag(C.model_from(sd1), u8, tc); print("SHORT", f"B3_{name}", 1.0, json.dumps(C.short(d1)), flush=True)
print(f"B3 total {time.time()-t00:.0f}s", flush=True)
