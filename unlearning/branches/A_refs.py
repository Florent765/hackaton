"""Branch A - cheap references. A0 unchanged export (pipeline check); A1 positive head scaling (argmax preserved,
calibration control); A2 pseudo-label CE ascent on the head only (+ L2 anchor), interpolated.
usage: python A_refs.py [A0|A1|A2]"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import time
import torch
import torch.nn.functional as F
import ul_lib as L

which = sys.argv[1] if len(sys.argv) > 1 else "A0"
u8 = L.load_images(); t0 = time.time(); tc = L.teacher_cache(u8); print(f"teacher cache {time.time()-t0:.0f}s", flush=True)
sd0 = L.original_state()
if which == "A0":
    m = L.load_original(); L.report("A0_orig", L.full_diag(m, u8, tc)); L.export(sd0, "A0_unchanged", {"branch": "A0"})
elif which == "A1":
    for s in (0.85, 0.7, 0.5):
        sd = dict(sd0); sd["head.weight"] = sd0["head.weight"] * s; sd["head.bias"] = sd0["head.bias"] * s
        m = L.build_model(); m.load_state_dict(sd); m = m.to(L.DEV)
        L.report(f"A1_scale{s}", L.full_diag(m, u8, tc)); L.export(sd, f"A1_headscale_{s}", {"branch": "A1", "scale": s})
elif which == "A2":
    m = L.load_original(); pl = tc["clean_0"].argmax(1)
    for p in m.parameters(): p.requires_grad_(False)
    m.head.weight.requires_grad_(True); m.head.bias.requires_grad_(True)
    w0 = m.head.weight.detach().clone(); b0 = m.head.bias.detach().clone()
    opt = torch.optim.SGD([m.head.weight, m.head.bias], lr=1e-3)
    with torch.no_grad():                                            # cache CLS features (frozen backbone)
        feats = torch.cat([m.forward_head(m.forward_features(L.prep(u8[i:i+128])), pre_logits=True).float() for i in range(0, len(u8), 128)])
    for step in range(200):
        idx = torch.randint(0, len(u8), (256,))
        lg = feats[idx] @ m.head.weight.T + m.head.bias
        loss = -F.cross_entropy(lg, pl[idx].to(L.DEV)) + 1.0 * ((m.head.weight - w0).pow(2).sum() + (m.head.bias - b0).pow(2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0: print("step", step, float(loss), flush=True)
    sdc = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    for a in (0.5, 1.0):
        sd = L.interpolate(sd0, sdc, a); mm = L.build_model(); mm.load_state_dict(sd); mm = mm.to(L.DEV)
        L.report(f"A2_head_ascent_a{a}", L.full_diag(mm, u8, tc)); L.export(sd, f"A2_head_ascent_a{a}", {"branch": "A2", "alpha": a})
print("A DONE")
