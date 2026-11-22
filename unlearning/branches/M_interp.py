"""Gentle versions of exported candidates: theta = theta0 + alpha (theta_c - theta0) for small alphas
(official feedback: the team's best 0.1476 came from a very gentle edit, KL(orig||sub) ~ 0.004 on forget).
usage: python M_interp.py <export_name> <a1,a2,...>"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul")
import torch
import ul_lib as L

name, alphas = sys.argv[1], [float(a) for a in sys.argv[2].split(",")]
sd0 = L.original_state(); sc = torch.load(L.HOME_OUT / f"{name}.pt", map_location="cpu", weights_only=True)
for a in alphas:
    L.export(L.interpolate(sd0, sc, a), f"G_{name}_a{a}", {"base": name, "alpha": a})
print("INTERP DONE")
