"""B1 Frankenstein stitching. Two descendants of theta0 that differ internally: A = theta0 + noise(seed a) healed by
KL(orig||.) on STRONG views, B = theta0 + noise(seed b) healed on MILD views. Stitch early blocks of A with late blocks
of B; the junction (block `cut`) is re-trained with KL to the original on strong views (class knowledge, never clean).
Controls: stitch(orig,orig) with the same junction perturbation+training; stitch(A,A) (same lineage, no cross).
Exported model = plain ViT-B/16 state dict (junction is just block `cut`).
usage: B1_stitch.py [--sigma 0.5] [--heal 500] [--jsteps 300] [--cut 6] [--lr 1e-4] [--jviews strong|both] [--tag p1]"""
import sys; sys.path.append("/p/home/jusers/jacquet2/jureca/ul"); import ul_lib as L   # append: ~/ul/inspect.py shadows stdlib
import argparse, copy, time
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--heal", type=int, default=500)
ap.add_argument("--jsteps", type=int, default=300); ap.add_argument("--cut", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--jviews", default="strong")
ap.add_argument("--tag", default="p1"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--export", default="")
A = ap.parse_args(); print(vars(A), flush=True)
T0 = time.time(); u8 = L.load_images(); N = len(u8)
orig = L.load_original(); [p.requires_grad_(False) for p in orig.parameters()]
p_cache = L.CACHE / "orig_logits.pt"
tc = torch.load(p_cache, weights_only=False) if p_cache.exists() else {f"{k}_0": L.logits(orig, u8, k, 0) for k in ("clean", "mild", "strong", "diag")}
print(f"teacher ready {time.time()-T0:.0f}s", flush=True)
sd0 = L.original_state()


def kl_t2(s, t, T=2.0):
    return F.kl_div(F.log_softmax(s / T, 1), F.log_softmax(t / T, 1), log_target=True, reduction="batchmean") * T * T


def perturb(m, blocks, sigma, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for b in blocks:
            for n, p in m.blocks[b].named_parameters():
                if p.ndim == 2:  # qkv / proj / fc1 / fc2 weights; norms, biases, LayerScale gammas untouched
                    p.add_((torch.randn(p.shape, generator=g) * p.std().item() * sigma).to(p.device))


def distill(m, blocks, steps, views, seed, lr):
    """Train m.blocks[blocks] (others frozen) with KL(orig||m) T=2 on forget-image views (never clean)."""
    [p.requires_grad_(False) for p in m.parameters()]
    params = [p for b in blocks for p in m.blocks[b].parameters()]; [p.requires_grad_(True) for p in params]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    for t in range(steps):
        m.train(); idx = torch.randint(0, N, (A.bs,), generator=gen)
        kind = views[t % len(views)]; x = L.view(L.prep(u8[idx]), kind, seed * 1_000_003 + t)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tl = orig(x).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = kl_t2(m(x).float(), tl)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if t % 100 == 0 or t == steps - 1:
            print(f"    t={t} {kind} kl={float(loss):.4f} {time.time()-T0:.0f}s", flush=True)
    [p.requires_grad_(False) for p in m.parameters()]; m.eval(); return m


def stitch(early, late, jseed, name):
    m = copy.deepcopy(late)
    with torch.no_grad():
        for b in range(A.cut):
            m.blocks[b].load_state_dict(early.blocks[b].state_dict())
        m.patch_embed.load_state_dict(early.patch_embed.state_dict())
        m.cls_token.copy_(early.cls_token); m.pos_embed.copy_(early.pos_embed)
        m.blocks[A.cut].load_state_dict(early.blocks[A.cut].state_dict())
    evaluate(m, f"{name}_nojunction")
    perturb(m, [A.cut], A.sigma, jseed)   # same junction perturbation in every stitch (incl. orig-orig control)
    views = ["strong"] if A.jviews == "strong" else ["strong", "mild"]
    print(f"  junction {name}", flush=True)
    return distill(m, [A.cut], A.jsteps, views, jseed, A.lr)


results = {}


def evaluate(m, name, alphas=()):
    d = L.full_diag(m, u8, tc); L.report(f"B1_{A.tag}_{name}", d)
    results[name] = d
    sd1 = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    for a in alphas:
        mi = L.build_model(); mi.load_state_dict(L.interpolate(sd0, sd1, a)); mi = mi.to(L.DEV).eval()
        L.report(f"B1_{A.tag}_{name}_a{a}", L.full_diag(mi, u8, tc))
        if name in A.export.split(","):
            L.export(L.interpolate(sd0, sd1, a), f"B1_{A.tag}_{name}_a{a}", vars(A))
    if name in A.export.split(","):
        L.export(sd1, f"B1_{A.tag}_{name}_a1", vars(A))
    return d


evaluate(orig, "orig")
allb = list(range(12))
print("descendant A (noise seed 1, heal strong)", flush=True)
dA = copy.deepcopy(orig); perturb(dA, allb, A.sigma, 1); evaluate(dA, "A_noised"); distill(dA, allb, A.heal, ["strong"], 11, A.lr)
evaluate(dA, "A", alphas=(0.5,))
print("descendant B (noise seed 2, heal mild)", flush=True)
dB = copy.deepcopy(orig); perturb(dB, allb, A.sigma, 2); distill(dB, allb, A.heal, ["mild"], 22, A.lr)
evaluate(dB, "B", alphas=(0.5,))
evaluate(stitch(dA, dB, 33, "stAB"), "stAB", alphas=(0.5,))
evaluate(stitch(orig, orig, 33, "stOO"), "stOO", alphas=(0.5,))
evaluate(stitch(dA, dA, 33, "stAA"), "stAA", alphas=(0.5,))
print(f"TOTAL {time.time()-T0:.0f}s", flush=True)
