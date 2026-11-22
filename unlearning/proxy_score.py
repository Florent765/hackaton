"""Local PROXY of the official unlearning score (NOT the official score).
Same formula as the server, with a declared surrogate gold:
  surrogate gold on a forget image = original model's mean softmax(T=2) over K=8 strong views of that image
  (how the original sees the image when it cannot rely on its exact memorised pixels).
  gold loss distribution = original CE over those strong views w.r.t. the clean pseudo-label (mean, std).
forget_proxy = loss_sim * pred_sim, computed on clean forget images AND on 'diag' views (held-out family, stands in
for the hidden altered copies), averaged over the two groups like the server.
  loss_sim = 1 - clip(d_sub/d_orig), d = |mean(L)-g_mean| + |std(L)-g_std|, L = CE(sub logits, pseudo-label)
  pred_sim = 1 - clip(mean KL(gold||sub)/mean KL(gold||orig)) at T=2
retain_proxy = agree_diag_mild * 1/(1+mean KL(orig||sub)) on mild+diag views of forget images (no control set exists)
proxy = forget_proxy * retain_proxy.   usage: python proxy_score.py name1 name2 ...  (files in ~/ul_out/<name>.pt)
Results appended to ~/ul_out/proxy_scores.tsv."""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import torch
import torch.nn.functional as F
import ul_lib as L

T = 2.0; K = 8
u8 = L.load_images(); orig = L.load_original(); n = len(u8)
cache = L.CACHE / "proxy_gold.pt"
if cache.exists():
    G = torch.load(cache, weights_only=False)
else:
    pl = L.logits(orig, u8, "clean").argmax(1)
    S = torch.stack([L.logits(orig, u8, "strong", seed=7000 + 101 * k) for k in range(K)])     # [K,n,100]
    gold_p = F.softmax(S / T, -1).mean(0)
    ce = torch.stack([F.cross_entropy(S[k], pl, reduction="none") for k in range(K)])
    G = {"pl": pl, "gold_p": gold_p, "g_mean": float(ce.mean()), "g_std": float(ce.std()),
         "orig": {k: L.logits(orig, u8, k) for k in ("clean", "diag", "mild")}}
    torch.save(G, cache)
print(f"surrogate gold loss mean/std {G['g_mean']:.3f}/{G['g_std']:.3f}", flush=True)


def kl(p_log_target, q_log):              # KL(P||Q) with P given as log-probs
    return (p_log_target.exp() * (p_log_target - q_log)).sum(1)


def group_scores(lg, lg0):
    pl, gp = G["pl"], G["gold_p"].clamp_min(1e-12).log()
    L_sub = F.cross_entropy(lg, pl, reduction="none"); L_0 = F.cross_entropy(lg0, pl, reduction="none")
    d = lambda Lx: abs(float(Lx.mean()) - G["g_mean"]) + abs(float(Lx.std()) - G["g_std"])
    loss_sim = 1 - min(max(d(L_sub) / d(L_0), 0), 1)
    k_sub = float(kl(gp, F.log_softmax(lg / T, 1)).mean()); k_0 = float(kl(gp, F.log_softmax(lg0 / T, 1)).mean())
    pred_sim = 1 - min(max(k_sub / k_0, 0), 1)
    return loss_sim, pred_sim, float(L_sub.mean()), float(L_sub.std())


rows = []
names = sys.argv[1:]
if not names or names == ["ALL"]:                     # score every export not yet in the table
    done = set()
    tsv = L.HOME_OUT / "proxy_scores.tsv"
    if tsv.exists():
        done = {l.split("\t")[0] for l in tsv.read_text().splitlines() if l.strip()}
    names = sorted(p.stem for p in L.HOME_OUT.glob("*.pt") if p.stem not in done)
print("scoring", len(names), "candidates", flush=True)
for name in names:
    m = L.build_model(); m.load_state_dict(torch.load(L.HOME_OUT / f"{name}.pt", map_location="cpu", weights_only=True), strict=True)
    m = m.to(L.DEV).eval()
    lg = {k: L.logits(m, u8, k) for k in ("clean", "diag", "mild")}
    ls_c, ps_c, lm_c, lsd_c = group_scores(lg["clean"], G["orig"]["clean"])
    ls_d, ps_d, _, _ = group_scores(lg["diag"], G["orig"]["diag"])
    forget = ((ls_c + ls_d) / 2) * ((ps_c + ps_d) / 2)
    klr = float(torch.cat([kl(F.log_softmax(G["orig"][k] / T, 1), F.log_softmax(lg[k] / T, 1)) for k in ("mild", "diag")]).mean())
    agr = float(torch.cat([(lg[k].argmax(1) == G["orig"][k].argmax(1)).float() for k in ("mild", "diag")]).mean())
    retain = agr * 1 / (1 + klr); score = forget * retain
    row = (name, round(score, 4), round(forget, 4), round(retain, 4), round((ls_c + ls_d) / 2, 3), round((ps_c + ps_d) / 2, 3),
           round(lm_c, 3), round(lsd_c, 3), round(klr, 4), round(agr, 4))
    rows.append(row)
    print("PROXY " + "\t".join(map(str, row)), flush=True)
    with open(L.HOME_OUT / "proxy_scores.tsv", "a") as f:
        f.write("\t".join(map(str, row)) + "\n")
    del m; torch.cuda.empty_cache()
print("cols: name proxy forget retain loss_sim pred_sim cleanCE_mean cleanCE_std KL_orig_sub agree")
print("PROXY DONE")
