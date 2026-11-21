"""Assemble a per-method combination of existing runs (rlib trains each method independently with the same seeds,
so taking method M from run X and the others from run Y equals a run where only M uses X's inputs).
usage: python assemble.py <new_exp> <base_exp> <method>=<exp> [<method>=<exp> ...] [--seeds 0 1 2]"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rlib as R  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("base"); ap.add_argument("subs", nargs="+")
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
a = ap.parse_args()
subs = dict(s.split("=") for s in a.subs)
out = R.OUT_ROOT / a.new; out.mkdir(parents=True, exist_ok=True)
for s in a.seeds:
    rec = torch.load(R.OUT_ROOT / a.base / f"seed{s}.pt", weights_only=False)
    for m, e in subs.items():
        other = torch.load(R.OUT_ROOT / e / f"seed{s}.pt", weights_only=False)
        assert other["seed"] == s and other["methods"][m]["names"] == rec["methods"][m]["names"]
        rec["methods"][m] = other["methods"][m]
    rec["exp"] = a.new; rec["notes"] = f"assembled: base {a.base}, {subs}"
    torch.save(rec, out / f"seed{s}.pt")
    print("wrote", out / f"seed{s}.pt")
print("---\nval_miou: 0.000000\nASSEMBLE DONE")
