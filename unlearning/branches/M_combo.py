"""Combination of complementary edits (local proxy suggested BC_qm improves predictions, B3 K-FAC improves the loss
distribution): theta = theta0 + (theta_A - theta0) + s * (theta_B - theta0). Both deltas come from exported
candidates (full state dicts in ~/ul_out). Exports each combination for proxy scoring.
usage: python M_combo.py <A_name> <B_name> <s1,s2,...> <tag>"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import torch
import ul_lib as L

A, B, ss, tag = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")], sys.argv[4]
sd0 = L.original_state()
sa = torch.load(L.HOME_OUT / f"{A}.pt", map_location="cpu", weights_only=True)
sb = torch.load(L.HOME_OUT / f"{B}.pt", map_location="cpu", weights_only=True)
na = sum(float((sa[k] - sd0[k]).norm() ** 2) for k in sd0) ** .5; nb = sum(float((sb[k] - sd0[k]).norm() ** 2) for k in sd0) ** .5
cos = sum(float(((sa[k] - sd0[k]) * (sb[k] - sd0[k])).sum()) for k in sd0) / (na * nb + 1e-12)
print(f"|dA| {na:.3f} |dB| {nb:.3f} cos(dA,dB) {cos:.4f}", flush=True)
for s in ss:
    sd = {k: sd0[k] + (sa[k] - sd0[k]) + s * (sb[k] - sd0[k]) for k in sd0}
    L.export(sd, f"MC_{tag}_s{s}", {"A": A, "B": B, "s": s})
print("COMBO DONE")
