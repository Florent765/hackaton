"""Branch B-JiT: faithful JiT (arXiv:2402.01401, code src/lipschitz.py::Lipschitz.modify_weight) + scope variants,
each with a matched calibration control (head scaling = global temperature, matched on clean pseudo-label CE).
Faithful details (from the code): SGD (no momentum/wd) over the chosen params, ONE pass over the forget set in order,
batch 128, one step per batch; loss = (1/N) sum_j sum_{i in batch} ||f(x_i) - f(x_i+xi_ij)||_2 / ||xi_ij||_2 with f = logits,
xi ~ N(0, sigma^2) added in normalised input space then clamped to the batch [min,max]; noisy branch under no_grad.
usage: python BJ_jit.py scope:lr:sigma:N [...]   scope in all|lastK|gamma|head"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import time
import torch
import ul_lib as L
import bjc

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
specs = sys.argv[1:] or ["all:0.01:0.5:25"]
ctx = bjc.Ctx()


def jit(scope, lr, sigma, N, bs=128, seed=0):
    torch.manual_seed(seed); m = L.load_original(); ps = bjc.params_for(m, scope)
    opt = torch.optim.SGD(ps, lr=lr); t0 = time.time(); losses = []
    for s in range(0, len(ctx.u8), bs):
        x = L.prep(ctx.u8[s:s + bs]); lo, hi = x.min(), x.max()
        out = m(x).float()                                              # clean branch, fp32/TF32, with grad
        loss = 0.0
        for _ in range(N):
            x2 = (x + torch.randn_like(x) * sigma).clamp(lo, hi)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out2 = m(x2).float()
            in_norm = (x - x2).flatten(1).norm(dim=1); out_norm = (out - out2).norm(dim=1)
            loss = loss + (out_norm / in_norm).sum().abs()
        loss = loss / N
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss))
        if not torch.isfinite(loss):
            print("NONFINITE loss, abort", flush=True); break
    print(f"  steps {len(losses)} loss first/mid/last {losses[0]:.3f}/{losses[len(losses)//2]:.3f}/{losses[-1]:.3f} "
          f"time {time.time()-t0:.0f}s", flush=True)
    return m


for spec in specs:
    scope, lr, sigma, N = spec.split(":"); lr, sigma, N = float(lr), float(sigma), int(N)
    name = f"BJ_{scope}_lr{lr}_s{sigma}_N{N}"; print("==", name, flush=True)
    m = jit(scope, lr, sigma, N)
    d = bjc.evaluate(m, ctx, name, {"drift": bjc.drift(m, ctx.sd0)})
    bjc.temp_control(ctx, d["clean"]["ce_clean_pl"], name + "_TEMPCTRL")
    bjc.save(m, name); del m; torch.cuda.empty_cache()
print("BJ DONE")
