"""B2 identity amnesia (adversarial). Train last blocks so that linear adversaries on CLS features at 3 depths
cannot tell WHICH forget image (instance id / 3500) an activation comes from, while KL(orig||model) on strong views
anchors class behaviour. Control: identical run with lam=0 (adversaries trained, not opposed).
Decisive test: NEW probes (linear + MLP, fresh view seeds) fitted after training; test on clean / diag / unseen strong.
usage: B2_identity.py [--steps 600] [--lam 1.0] [--first 6] [--lr 3e-5] [--anchor strong|both] [--tag x] [--export 0]"""
import sys; sys.path.append("/p/home/jusers/jacquet2/jureca/ul"); import ul_lib as L   # append: ~/ul/inspect.py shadows stdlib
import argparse, copy, time
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lam", default="1.0")
ap.add_argument("--first", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--anchor", default="strong"); ap.add_argument("--tag", default="p1"); ap.add_argument("--export", type=int, default=0)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--controls", type=int, default=1)
ap.add_argument("--alphas", default="0.5"); ap.add_argument("--probes", type=int, default=1); ap.add_argument("--name", default="")
ap.add_argument("--advmode", default="id")  # id | random (frozen random probes = non-identity control)
A = ap.parse_args(); print(vars(A), flush=True)
torch.manual_seed(0); T0 = time.time()
u8 = L.load_images(); N = len(u8)
orig = L.load_original(); [p.requires_grad_(False) for p in orig.parameters()]
p_cache = L.CACHE / "orig_logits.pt"
tc = torch.load(p_cache, weights_only=False) if p_cache.exists() else {f"{k}_0": L.logits(orig, u8, k, 0) for k in ("clean", "mild", "strong", "diag")}
print(f"teacher ready {time.time()-T0:.0f}s (cache={p_cache.exists()})", flush=True)
DEPTHS = (7, 9, "norm")


def feats_hook(m):
    store = {}
    hs = [m.blocks[d].register_forward_hook(lambda mod, i, o, d=d: store.__setitem__(d, o[:, 0])) for d in DEPTHS if d != "norm"]
    hs.append(m.norm.register_forward_hook(lambda mod, i, o: store.__setitem__("norm", o[:, 0])))
    return store, hs


@torch.no_grad()
def extract(m, kind, seed, bs=256):
    m.eval(); store, hs = feats_hook(m); out = {d: [] for d in DEPTHS}
    for s in range(0, N, bs):
        with torch.autocast("cuda", dtype=torch.float16):
            m(L.view(L.prep(u8[s:s + bs]), kind, seed + s))
        for d in DEPTHS: out[d].append(store[d].float())
    for h in hs: h.remove()
    return {d: torch.cat(v) for d, v in out.items()}


def make_probe(kind="lin"):
    if kind == "lin":
        return nn.Sequential(nn.LayerNorm(768, elementwise_affine=False), nn.Linear(768, N)).to(L.DEV)
    return nn.Sequential(nn.LayerNorm(768, elementwise_affine=False), nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, N)).to(L.DEV)


def fit_probe(X, y, kind="lin", steps=400, lr=2e-3, seed=0):
    torch.manual_seed(seed); p = make_probe(kind); opt = torch.optim.Adam(p.parameters(), lr=lr)
    for t in range(steps):
        b = torch.randint(0, len(X), (2048,), device=L.DEV)
        loss = F.cross_entropy(p(X[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    return p


def kl_t2(student, teacher, T=2.0):
    return F.kl_div(F.log_softmax(student / T, 1), F.log_softmax(teacher / T, 1), log_target=True, reduction="batchmean") * T * T


def uniform_ce(z):  # CE between adversary softmax and uniform target -> minimised when adversary is maximally unsure
    return (torch.logsumexp(z, 1) - z.mean(1)).mean()


def train(lam, seed):
    torch.manual_seed(seed)
    m = copy.deepcopy(orig); [p.requires_grad_(False) for p in m.parameters()]
    params = [p for b in m.blocks[A.first:] for p in b.parameters()] + list(m.norm.parameters())
    [p.requires_grad_(True) for p in params]
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0)
    # adversary warm start on original features (4 strong + 2 mild views)
    F0 = [extract(orig, "strong", 100 + 7919 * v) for v in range(4)] + [extract(orig, "mild", 200 + 7919 * v) for v in range(2)]
    y_all = torch.arange(N, device=L.DEV).repeat(len(F0))
    if A.advmode == "random":
        torch.manual_seed(seed); advs = {d: make_probe("lin") for d in DEPTHS}
        with torch.no_grad():  # scale so the random probe logits have a spread comparable to trained ones
            for a in advs.values(): a[1].weight.mul_(30.0)
    else:
        advs = {d: fit_probe(torch.cat([f[d] for f in F0]), y_all, "lin", 300, seed=seed) for d in DEPTHS}
    aopt = torch.optim.Adam([p for a in advs.values() for p in a.parameters()], lr=1e-3 if A.advmode == "id" else 0.0)
    store, hs = feats_hook(m); log = []
    for t in range(A.steps):
        m.train(); idx = torch.randint(0, N, (A.bs,)); y = idx.to(L.DEV); x = L.prep(u8[idx])
        xs = L.view(x, "strong", 10_000 + seed * 100_000 + t); xm = L.view(x, "mild", 20_000 + seed * 100_000 + t)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ts = orig(xs).float(); tm = orig(xm).float() if A.anchor == "both" else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ls = m(xs).float(); fs = {d: store[d].float() for d in DEPTHS}
            lm = m(xm).float(); fm = {d: store[d].float() for d in DEPTHS}
        loss_kl = kl_t2(ls, ts) + (kl_t2(lm, tm) if tm is not None else 0.0)
        # adversary step (detached features)
        aloss = sum(F.cross_entropy(advs[d](f[d].detach()), y) for f in (fs, fm) for d in DEPTHS)
        aopt.zero_grad(); aloss.backward(); aopt.step()
        # model step
        with torch.no_grad():
            acc = sum((advs[d](fs[d]).argmax(1) == y).float().mean().item() for d in DEPTHS) / len(DEPTHS)
        loss_id = sum(uniform_ce(advs[d](f[d])) for f in (fs, fm) for d in DEPTHS) / (2 * len(DEPTHS))
        loss = loss_kl + lam * loss_id
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if t % 100 == 0 or t == A.steps - 1:
            print(f"  lam={lam} t={t} kl={float(loss_kl):.4f} id={float(loss_id):.3f} adv_ce={float(aloss)/6:.3f} adv_acc={acc:.3f} {time.time()-T0:.0f}s", flush=True)
    for h in hs: h.remove()
    m.eval(); return m


def fresh_probe_test(m, name):
    """NEW adversaries never used in optimisation; fresh view seeds; test on clean / diag / unseen strong views."""
    tr = [extract(m, "strong", 5000 + 104729 * v) for v in range(4)] + [extract(m, "mild", 6000 + 104729 * v) for v in range(2)]
    te = {"clean": extract(m, "clean", 0), "diag": extract(m, "diag", 0), "strong_new": extract(m, "strong", 999_999)}
    y = torch.arange(N, device=L.DEV); res = {}
    for d in DEPTHS:
        X = torch.cat([f[d] for f in tr]); Y = y.repeat(len(tr))
        for kind in ("lin", "mlp"):
            p = fit_probe(X, Y, kind, 400, seed=123)
            with torch.no_grad():
                for k, f in te.items():
                    res[f"{d}_{kind}_{k}"] = round(float((p(f[d]).argmax(1) == y).float().mean()), 4)
    L.report(f"B2_{A.tag}_{name}_fresh_probe_id_acc", res); return res


def evaluate(m, name):
    d = L.full_diag(m, u8, tc); L.report(f"B2_{A.tag}_{name}", d); return d


sd0 = L.original_state()
evaluate(orig, "orig")
if A.probes: fresh_probe_test(orig, "orig")
runs = [(f"adv_l{l}" if "," in A.lam else "adv", float(l)) for l in A.lam.split(",")] + ([("ctrl_lam0", 0.0)] if A.controls else [])
for name, lam in runs:
    m = train(lam, seed=1)
    evaluate(m, name)
    if A.probes: fresh_probe_test(m, name)
    sd1 = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    for a in [float(v) for v in A.alphas.split(",")]:
        mi = L.build_model(); mi.load_state_dict(L.interpolate(sd0, sd1, a)); mi = mi.to(L.DEV).eval()
        evaluate(mi, f"{name}_a{a}")
        if A.export and name.startswith("adv"):
            L.export(L.interpolate(sd0, sd1, a), (f"{A.name}_{name}" if A.name else f"B2_{A.tag}_{name}") + f"_a{a}", vars(A))
    if A.export and name.startswith("adv"):
        L.export(sd1, (f"{A.name}_{name}" if A.name else f"B2_{A.tag}_{name}") + "_a1", vars(A))
    del m; torch.cuda.empty_cache()
print(f"TOTAL {time.time()-T0:.0f}s", flush=True)
