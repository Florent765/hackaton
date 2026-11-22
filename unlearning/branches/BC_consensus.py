"""Branch C: consensus distillation. Frozen teacher = original over K views (mild seeds 1..K, strong seeds 1..K; never
'diag'). Target on forget images q_beta = (1-b) p_orig(x) + b q(x) at T=2 (q: prob-average or logit-average of view
predictions), loss T^2 KL(q_beta || p_student) on clean (or augmented) forget images + lam * T^2 KL(p_orig || p_student)
on anchors (mild views of forget images, or CutMix of forget pairs as pseudo-retain, or none).
Quantile-matching variant 'qm': per-image target CE c_i = quantile map of consensus-CE rank onto the pooled view-CE
distribution (vs clean pseudo-label); target t_i[pl]=exp(-c_i), rest of mass shaped like q; KL at T=1.
usage: python BC_consensus.py target:beta:scope:anchor:lam:fin:epochs:lr [...]
   target prob|logit|qm  (qm: beta = source mix|strong)   anchor mild|cm|none   fin clean|aug"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import time
import torch
import torch.nn.functional as F
import ul_lib as L
import bjc

T = 2.0; K = 8; BS = 64
specs = sys.argv[1:]
ctx = bjc.Ctx(); u8 = ctx.u8; n = len(u8); pl = ctx.pl.to(L.DEV)
orig = L.load_original()
for p in orig.parameters():
    p.requires_grad_(False)

tp = bjc.SAVE / "view_logits_K8.pt"
if tp.exists():
    VL = torch.load(tp)
else:
    t0 = time.time(); VL = {}
    for fam in ("mild", "strong"):
        VL[fam] = torch.stack([L.logits(orig, u8, fam, seed=1000 * k + 7) for k in range(1, K + 1)])   # [K,n,100]
    torch.save(VL, tp); print(f"view logits {time.time()-t0:.0f}s", flush=True)
allv = torch.cat([VL["mild"], VL["strong"]])                                                          # [2K,n,100]
z0 = ctx.tc["clean_0"]
q_prob = F.softmax(allv / T, -1).mean(0); q_logit = F.softmax(allv.mean(0) / T, -1); p0 = F.softmax(z0 / T, -1)
for nm, q in (("prob", q_prob), ("logit", q_logit)):
    ce_q = -torch.log(q.gather(1, ctx.pl[:, None]).squeeze(1).clamp_min(1e-12))
    print(f"teacher {nm}@T2: conf {q.max(1).values.mean():.4f} pl-prob {q.gather(1, ctx.pl[:, None]).mean():.4f} "
          f"argmax!=pl {(q.argmax(1) != ctx.pl).float().mean():.4f} ce_pl mean/sd {ce_q.mean():.3f}/{ce_q.std():.3f} "
          f"JS-ish KL(q||p0) {(q * (q.clamp_min(1e-12).log() - p0.clamp_min(1e-12).log())).sum(1).mean():.4f}", flush=True)
for nm, q in (("prob1", F.softmax(allv, -1).mean(0)), ("strong1", F.softmax(VL["strong"], -1).mean(0))):
    ce_q = -torch.log(q.gather(1, ctx.pl[:, None]).squeeze(1).clamp_min(1e-12))
    print(f"teacher {nm}@T1: conf {q.max(1).values.mean():.4f} argmax!=pl {(q.argmax(1) != ctx.pl).float().mean():.4f} "
          f"ce_pl mean/sd {ce_q.mean():.3f}/{ce_q.std():.3f} q05/50/95 {[round(float(v), 3) for v in torch.quantile(ce_q, torch.tensor([.05, .5, .95]))]}", flush=True)
for fam in ("mild", "strong"):
    cev = F.cross_entropy(VL[fam].flatten(0, 1), ctx.pl.repeat(K), reduction="none")
    print(f"orig {fam} views (K={K}) CE vs clean pl mean/sd {cev.mean():.3f}/{cev.std():.3f} "
          f"agree {(VL[fam].argmax(-1) == ctx.pl).float().mean():.4f}", flush=True)


def qm_target(source):
    pool = torch.cat([VL["mild"], VL["strong"]]) if source == "mix" else VL["strong"]
    ce_pool = F.cross_entropy(pool.flatten(0, 1), ctx.pl.repeat(len(pool)), reduction="none").sort().values
    q1 = F.softmax(allv, -1).mean(0); r = -torch.log(q1.gather(1, ctx.pl[:, None]).squeeze(1).clamp_min(1e-12))
    rank = r.argsort().argsort().float() / (n - 1)
    c = ce_pool[(rank * (len(ce_pool) - 1)).long()]
    ppl = torch.exp(-c); rest = q1.clone(); rest.scatter_(1, ctx.pl[:, None], 0.0)
    rest = rest / rest.sum(1, keepdim=True).clamp_min(1e-12)
    t = rest * (1 - ppl)[:, None]; t.scatter_(1, ctx.pl[:, None], ppl[:, None])
    print(f"qm[{source}] target CE mean/sd {c.mean():.3f}/{c.std():.3f} q05/50/95 "
          f"{[round(float(v), 3) for v in torch.quantile(c, torch.tensor([.05, .5, .95]))]}", flush=True)
    return t


def anchor_batch(kind, g):
    idx = torch.randint(0, n, (BS,), generator=g)
    x = L.prep(u8[idx])
    if kind == "mild":
        return L.view(x, "mild", int(torch.randint(0, 10**6, (1,), generator=g)))
    j = torch.randint(0, n, (BS,), generator=g); x2 = L.prep(u8[j])
    s = int(224 * (0.3 + 0.4 * torch.rand(1, generator=g).item())); a, b = torch.randint(0, 224 - s + 1, (2,), generator=g).tolist()
    x = x.clone(); x[:, :, a:a + s, b:b + s] = x2[:, :, a:a + s, b:b + s]; return x


def run(target, beta, scope, anchor, lam, fin, epochs, lr, seed=0):
    if target == "qm":
        tgt, Tt = qm_target(beta), 1.0
    elif target in ("prob1", "strong1"):                                   # consensus at T=1 (loss at T=1)
        b = float(beta); src = allv if target == "prob1" else VL["strong"]
        tgt, Tt = (1 - b) * F.softmax(z0, -1) + b * F.softmax(src, -1).mean(0), 1.0
    else:
        b = float(beta); q = q_prob if target == "prob" else q_logit
        tgt, Tt = (1 - b) * p0 + b * q, T
    tgt = tgt.to(L.DEV)
    m = L.load_original(); ps = bjc.params_for(m, scope); m.train()
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed); t0 = time.time(); hist = []
    steps = epochs * (n // BS)
    for step in range(steps):
        idx = torch.randint(0, n, (BS,), generator=g)
        x = L.prep(u8[idx])
        if fin == "aug":
            r = int(torch.randint(0, 3, (1,), generator=g)); sd = int(torch.randint(0, 10**6, (1,), generator=g))
            x = x if r == 0 else L.view(x, "mild" if r == 1 else "strong", sd)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs = m(x).float()
            lf = F.kl_div(F.log_softmax(zs / Tt, 1), tgt[idx], reduction="batchmean") * Tt * Tt
            la = torch.zeros((), device=L.DEV)
            if anchor != "none" and lam > 0:
                xa = anchor_batch(anchor, g)
                with torch.no_grad():
                    za0 = orig(xa).float()
                za = m(xa).float()
                la = F.kl_div(F.log_softmax(za / T, 1), F.log_softmax(za0 / T, 1), log_target=True, reduction="batchmean") * T * T
            loss = lf + lam * la
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); hist.append((float(lf), float(la)))
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step}/{steps} lf {hist[-1][0]:.4f} la {hist[-1][1]:.4f} {time.time()-t0:.0f}s", flush=True)
    m.eval(); return m


for spec in specs:
    target, beta, scope, anchor, lam, fin, epochs, lr = spec.split(":")
    name = f"BC_{target}_b{beta}_{scope}_{anchor}{lam}_{fin}_e{epochs}_lr{lr}"; print("==", name, flush=True)
    m = run(target, beta, scope, anchor, float(lam), fin, int(epochs), float(lr))
    d = bjc.evaluate(m, ctx, name, {"drift": bjc.drift(m, ctx.sd0)})
    bjc.temp_control(ctx, d["clean"]["ce_clean_pl"], name + "_TEMPCTRL")
    bjc.save(m, name); del m; torch.cuda.empty_cache()
print("BC DONE")
