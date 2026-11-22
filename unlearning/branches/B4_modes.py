"""B4 - change solution, keep competence. Re-initialise the MLPs of a few late blocks and re-fit them by distillation
from the original on STRONG/MILD views only (never clean forget images). Controls: 'perm' (hidden-unit permutation,
function-identical) and '<v>c' (same distillation but 1/3 of each batch are clean forget images).
usage: python B4_modes.py "<variant>:<blocks>;..." [steps]   variants: perm | mlp (strong+mild) | mlps (strong only)
       | mlpc (strong+mild+clean control)    e.g. "mlps:10,11;mlpc:10,11" """
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import json
import math
import time
import torch
import ul_lib as L
import B_common as C

configs = [(c.split(":")[0], [int(b) for b in c.split(":")[1].split(",")]) for c in sys.argv[1].split(";")]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
BS = 96
u8, tc, sd0, fit, held = C.setup()
teacher = C.model_from(sd0)
for p in teacher.parameters(): p.requires_grad_(False)
def rel_dist(sd):
    mlp_keys = [f"blocks.{b}.mlp.{n}.{t}" for b in BLOCKS for n in ("fc1", "fc2") for t in ("weight", "bias")]
    num = sum(float((sd[k].float() - sd0[k].float()).norm() ** 2) for k in mlp_keys)
    den = sum(float(sd0[k].float().norm() ** 2) for k in mlp_keys)
    return math.sqrt(num / den)


for v, BLOCKS in configs:
    t0 = time.time(); tag = f"{v}_b{'-'.join(map(str, BLOCKS))}"
    if v == "perm":
        sd = dict(sd0); g = torch.Generator().manual_seed(0)
        for b in BLOCKS:
            P = torch.randperm(sd0[f"blocks.{b}.mlp.fc1.weight"].shape[0], generator=g); pre = f"blocks.{b}.mlp."
            sd[pre + "fc1.weight"] = sd0[pre + "fc1.weight"][P]; sd[pre + "fc1.bias"] = sd0[pre + "fc1.bias"][P]
            sd[pre + "fc2.weight"] = sd0[pre + "fc2.weight"][:, P]
        print(f"perm rel_dist {rel_dist(sd):.3f}", flush=True)
        d = L.full_diag(C.model_from(sd), u8, tc); L.report("B4_perm", d); print("SHORT perm", json.dumps(C.short(d)), flush=True)
        continue
    with_clean = v == "mlpc"; strong_only = v == "mlps"
    m = C.model_from(sd0); torch.manual_seed(1)
    for b in BLOCKS:                                    # timm-style re-init of the MLP
        for lin in (m.blocks[b].mlp.fc1, m.blocks[b].mlp.fc2):
            torch.nn.init.trunc_normal_(lin.weight, std=0.02); torch.nn.init.zeros_(lin.bias)
    for p in m.parameters(): p.requires_grad_(False)
    params = [p for b in BLOCKS for p in m.blocks[b].mlp.parameters()]
    for p in params: p.requires_grad_(True)
    d = L.full_diag(m, u8, tc, idx=held[:500]); print("SHORT", v, "reinit_noFit", json.dumps(C.short(d)), flush=True)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    m.train()
    for step in range(STEPS):
        idx = torch.randint(0, len(u8), (BS,)); x = L.prep(u8[idx]); s = 1000 + step
        parts = [L.view(x[:BS // 2], "strong", s), L.view(x[BS // 2:], "mild", s)]
        if strong_only:
            parts = [L.view(x[:BS // 2], "strong", s), L.view(x[BS // 2:], "strong", s + 500000)]
        if with_clean:
            parts = [parts[0][: BS // 3], parts[1][: BS // 3], x[: BS // 3]]
        xb = torch.cat(parts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                lt = teacher(xb).float()
            loss = C.kl_t(lt, m(xb).float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"{v} step {step} loss {loss.item():.4f} t {time.time()-t0:.0f}s", flush=True)
    m.eval(); sd1 = {k: t.detach().cpu().float() for k, t in m.state_dict().items()}
    print(f"{v} rel_dist {rel_dist(sd1):.3f} train_time {time.time()-t0:.0f}s", flush=True)
    C.save_sd(sd1, f"B4_{tag}_s{STEPS}")
    C.eval_grid(sd0, sd1, f"B4_{tag}", u8, tc, alphas=(0.25, 0.5, 1.0))
    print(f"{v} total {time.time()-t0:.0f}s", flush=True)
    del m, opt; torch.cuda.empty_cache()
