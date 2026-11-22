"""B5 - evolving population of amnesic models (NSGA-II-lite). Genes: per-block log-scales of LayerScale gammas (ls1, ls2
x 12), head temperature (log-scale of head W,b), bias shift on the 45 pseudo-label classes, interpolation alpha (0..2) toward
an auxiliary saved model (arg 3: a B4 re-fit or the B3 K-FAC removal step). Objectives (all minimised, FIT images only):
 f1 = KL_T2(orig on strong views (2 seeds averaged) || model on CLEAN)   -> clean behaves like non-memorised views
 f2 = KL_T2(orig || model) on mild views                                 -> preservation
 f3 = KL_T2(orig || model) on strong views (other seed)                  -> preservation on 'generic-looking' inputs
Held-out check (HELD images, 'diag' views, never used for selection): f1_held, KL_diag, clean CE/conf.
usage: python B5_evo.py [pop] [gens] [b4_name]"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import json
import time
import torch
import torch.nn.functional as F
import ul_lib as L
import B_common as C

POP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
B4N = sys.argv[3] if len(sys.argv) > 3 else "B4_mlp_b1011_s800"
t00 = time.time()
u8, tc, sd0, fit, held = C.setup()
NF, NH = 600, 500
fi = fit[:NF].tolist(); hi = held[:NH].tolist()
try:
    sdB4 = C.load_sd(B4N); print("using B4 model", B4N, flush=True)
except Exception as e:
    sdB4 = None; print("no B4 model:", e, flush=True)
m = C.model_from(sd0)
gpu0 = {k: v.to(L.DEV) for k, v in sd0.items()}
gpuB4 = {k: v.to(L.DEV) for k, v in sdB4.items()} if sdB4 is not None else None
fcls = tc["clean_0"].argmax(1).unique().to(L.DEV)
NG = 24 + 3


def lp(lg, T=2.0):
    return F.log_softmax(lg / T, 1)


# targets (original model)
torch.manual_seed(0)
t_strong = torch.stack([lp(L.logits(m, u8, "strong", s, idx=fi)).exp() for s in (11, 12)]).mean(0)
t_mild = L.logits(m, u8, "mild", 21, idx=fi); t_str3 = L.logits(m, u8, "strong", 31, idx=fi)
th_strong = lp(L.logits(m, u8, "strong", 11, idx=hi)).exp(); th_diag = L.logits(m, u8, "diag", 0, idx=hi)   # same idx-call as evaluation => identical views
kl_p = lambda q, lg: float((q * (q.clamp_min(1e-12).log() - lp(lg))).sum(1).mean())     # KL(q || softmax(lg/2))
kl_l = lambda lt, lg: kl_p(lp(lt).exp(), lg)


@torch.no_grad()
def apply(gn):
    sd = gpu0 if (gpuB4 is None or gn[26] <= 0) else {k: gpu0[k] + float(gn[26]) * (gpuB4[k] - gpu0[k]) if gpu0[k].is_floating_point() else gpu0[k] for k in gpu0}
    sd = dict(sd)
    for b in range(12):
        sd[f"blocks.{b}.ls1.gamma"] = sd[f"blocks.{b}.ls1.gamma"] * float(torch.exp(gn[2 * b]))
        sd[f"blocks.{b}.ls2.gamma"] = sd[f"blocks.{b}.ls2.gamma"] * float(torch.exp(gn[2 * b + 1]))
    s = float(torch.exp(gn[24])); sd["head.weight"] = sd["head.weight"] * s
    hb = sd["head.bias"] * s; hb = hb.clone(); hb[fcls] += float(gn[25]); sd["head.bias"] = hb
    return sd


@torch.no_grad()
def evaluate(gn, heldout=False):
    m.load_state_dict(apply(gn), strict=True); m.eval()
    if heldout:
        lc = L.logits(m, u8, "clean", 0, idx=hi); ld = L.logits(m, u8, "diag", 0, idx=hi)
        pl = tc["clean_0"][hi].argmax(1); ce = F.cross_entropy(lc, pl, reduction="none")
        return {"f1_held": round(kl_p(th_strong, lc), 4), "kl_diag": round(kl_l(th_diag, ld), 4),
                "ce_clean": round(float(ce.mean()), 4), "ce_sd": round(float(ce.std()), 4),
                "conf_clean": round(float(F.softmax(lc, 1).max(1).values.mean()), 4),
                "chg_clean": round(float((lc.argmax(1) != pl).float().mean()), 4),
                "chg_diag": round(float((ld.argmax(1) != th_diag.argmax(1)).float().mean()), 4)}
    lc = L.logits(m, u8, "clean", 0, idx=fi); lm = L.logits(m, u8, "mild", 21, idx=fi); ls = L.logits(m, u8, "strong", 31, idx=fi)
    return [kl_p(t_strong, lc), kl_l(t_mild, lm), kl_l(t_str3, ls)]


def nd_sort(F_):
    n = len(F_); dom = lambda a, b: all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
    rank = [0] * n; rem = set(range(n)); r = 0
    while rem:
        front = [i for i in rem if not any(dom(F_[j], F_[i]) for j in rem if j != i)]
        for i in front: rank[i] = r
        rem -= set(front); r += 1
    return rank


def crowd(F_, idx):
    d = {i: 0.0 for i in idx}
    for k in range(3):
        s = sorted(idx, key=lambda i: F_[i][k]); d[s[0]] = d[s[-1]] = 1e9; span = F_[s[-1]][k] - F_[s[0]][k] + 1e-12
        for a, b, c in zip(s, s[1:], s[2:]): d[b] += (F_[c][k] - F_[a][k]) / span
    return d


g = torch.Generator().manual_seed(0)
def mutate(gn, sig=1.0):
    c = gn.clone(); mask = torch.rand(NG, generator=g) < 0.3
    step = torch.randn(NG, generator=g) * torch.tensor([0.15] * 24 + [0.15, 0.4, 0.2]) * sig
    c[mask] += step[mask]; c[26] = c[26].clamp(0, 2) if gpuB4 is not None else 0.0; c[25] = c[25].clamp(max=0.0)
    return c


base = torch.zeros(NG)
f_base = evaluate(base); print("orig objectives", [round(x, 4) for x in f_base], "held", evaluate(base, True), flush=True)
pop = [base] + [mutate(base, 2.0) for _ in range(POP - 1)]
fits = [evaluate(x) for x in pop]
for gen in range(GENS):
    t0 = time.time(); kids = []
    for _ in range(POP):
        a, b = torch.randint(0, len(pop), (2,), generator=g).tolist()
        cm = torch.rand(NG, generator=g) < 0.5; kid = torch.where(cm, pop[a], pop[b]); kids.append(mutate(kid))
    allp = pop + kids; allf = fits + [evaluate(x) for x in kids]
    rk = nd_sort(allf); order = []
    for r in range(max(rk) + 1):
        idx = [i for i in range(len(allp)) if rk[i] == r]
        if len(order) + len(idx) <= POP: order += idx
        else:
            cd = crowd(allf, idx); order += sorted(idx, key=lambda i: -cd[i])[: POP - len(order)]; break
    pop = [allp[i] for i in order]; fits = [allf[i] for i in order]
    f0 = [i for i in range(len(pop)) if nd_sort(fits)[i] == 0]
    print(f"gen {gen} {time.time()-t0:.0f}s front={len(f0)} best_f1={min(f[0] for f in fits):.3f} "
          f"f2@bestf1={fits[min(range(len(fits)), key=lambda i: fits[i][0])][1]:.3f}", flush=True)

rk = nd_sort(fits); front = sorted([i for i in range(len(pop)) if rk[i] == 0], key=lambda i: fits[i][0])
rows = []
for i in front:
    h = evaluate(pop[i], True); gn = pop[i]
    rows.append({"i": i, "f1": round(fits[i][0], 4), "f2_mild": round(fits[i][1], 4), "f3_strong": round(fits[i][2], 4), **h,
                 "temp": round(float(torch.exp(gn[24])), 3), "fbias": round(float(gn[25]), 3), "b4a": round(float(gn[26]), 3),
                 "ls_min": round(float(gn[:24].min()), 3), "ls_max": round(float(gn[:24].max()), 3)})
    print("FRONT", json.dumps(rows[-1]), flush=True)
xs = torch.tensor([[r["f1"], r["f1_held"], r["f2_mild"], r["kl_diag"]] for r in rows])
if len(rows) > 2:
    rkc = lambda v: v.argsort().argsort().float()
    sp = lambda a, b: float(torch.corrcoef(torch.stack([rkc(a), rkc(b)]))[0, 1])
    print(f"rank-corr fit-f1 vs held-f1 {sp(xs[:,0], xs[:,1]):.3f}; fit-f2 vs held-KLdiag {sp(xs[:,2], xs[:,3]):.3f}", flush=True)
# pick: lowest f1 with f2_mild <= budget, for 2 budgets; full diag on alpha grid (alpha scales genes, i.e. the edit)
for bud in (0.1, 0.3):
    ok = [r for r in rows if r["f2_mild"] <= bud]
    if not ok: print("no candidate for budget", bud); continue
    r = min(ok, key=lambda r: r["f1"]); gn = pop[r["i"]]; print("PICK", bud, json.dumps(r), "genes", [round(float(x), 3) for x in gn], flush=True)
    sd1 = {k: v.cpu() for k, v in apply(gn).items()}; C.save_sd(sd1, f"B5_pick_b{bud}")
    L.export(sd1, f"B5_pick_b{bud}", {"genes": [round(float(x), 4) for x in gn], "aux": B4N, **r})
    C.eval_grid(sd0, sd1, f"B5_pick_b{bud}", u8, tc)
print(f"B5 total {time.time()-t00:.0f}s", flush=True)
