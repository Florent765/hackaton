"""Interpolate original -> saved branch state (bjc.SAVE/<name>.pt) and evaluate; optionally export.
usage: python BX_interp.py name:a1,a2,... [...] [--export name:a1,...]   (exports only items after --export)"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import torch
import ul_lib as L
import bjc

args = sys.argv[1:]; ex = args.index("--export") if "--export" in args else len(args)
ctx = bjc.Ctx()
for i, item in enumerate(args):
    if item == "--export":
        continue
    name, al = item.split(":"); sd1 = torch.load(bjc.SAVE / f"{name}.pt", map_location="cpu")
    for a in [float(v) for v in al.split(",")]:
        sd = L.interpolate(ctx.sd0, sd1, a); m = L.build_model(); m.load_state_dict(sd); m = m.to(L.DEV).eval()
        tag = f"{name}_a{a}"; d = bjc.evaluate(m, ctx, tag)
        bjc.temp_control(ctx, d["clean"]["ce_clean_pl"], tag + "_TEMPCTRL")
        if i > ex:
            L.export(sd, tag, {"branch": "BJC", "source": name, "alpha": a, "diag": d})
        del m; torch.cuda.empty_cache()
print("BX DONE")
