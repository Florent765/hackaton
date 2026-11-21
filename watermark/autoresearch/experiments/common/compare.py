"""Paired comparison of research experiments (light CPU work: OK on a login node).
  python compare.py <ref_exp> <cand_exp> [--seeds 0 1 2] [--tsv line_label]
Merges the requested seeds of each experiment (mean of probabilities), evaluates the three threshold policies
(fixed 0.5 / exploratory OOF two-thresholds / confirmatory inner-calibrated two-thresholds), and reports per-image
paired deltas: mean, median, wins/losses/ties, stratified-by-method bootstrap 95% CI, by method and coverage bin.
Only seeds present in BOTH experiments are used (pairing)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rlib as R  # noqa: E402

BINS = ((0, .05), (.05, .5), (.5, .95), (.95, 1.01))


def load(exp, seeds):
    d = R.OUT_ROOT / exp
    have = sorted(int(p.stem[4:]) for p in d.glob("seed*.pt"))
    use = [s for s in (seeds or have) if s in have]
    return use, [torch.load(d / f"seed{s}.pt", weights_only=False) for s in use]


def flat(summary, key):
    v, m, c = [], [], []
    for method, pi in summary["per_image"].items():
        v += pi[key].tolist(); m += [method] * len(pi[key]); c += pi["cov"].tolist()
    return np.array(v), np.array(m), np.array(c)


def boot_ci(delta, methods, n=2000, seed=0):
    rng = np.random.RandomState(seed); groups = [np.nonzero(methods == u)[0] for u in np.unique(methods)]
    stats = []
    for _ in range(n):
        idx = np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups])
        stats.append(delta[idx].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("cand")
    ap.add_argument("--seeds", type=int, nargs="*"); ap.add_argument("--json", default="")
    a = ap.parse_args()
    s_ref, r_ref = load(a.ref, a.seeds); s_c, r_c = load(a.cand, a.seeds)
    common = sorted(set(s_ref) & set(s_c))
    r_ref = [r for r in r_ref if r["seed"] in common]; r_c = [r for r in r_c if r["seed"] in common]
    assert common, "no common seeds"
    S_ref, S_c = R.summarize(r_ref), R.summarize(r_c)
    for m in S_ref["per_image"]:
        assert S_ref["per_image"][m]["names"] == S_c["per_image"][m]["names"], "image order differs"
    report = {"ref": a.ref, "cand": a.cand, "seeds": common, "policies": {}}
    print(f"{a.cand} vs {a.ref}  seeds {common}")
    for key in ("fixed05", "explor", "confirm"):
        vr, mth, cov = flat(S_ref, key); vc, _, _ = flat(S_c, key); d = vc - vr
        lo, hi = boot_ci(d, mth)
        row = {"ref": float(vr.mean()), "cand": float(vc.mean()), "delta": float(d.mean()), "median": float(np.median(d)),
               "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ci95": [lo, hi],
               "by_method": {u: float(d[mth == u].mean()) for u in np.unique(mth)},
               "by_cov": {f"[{l},{h})": [float(d[(cov >= l) & (cov < h)].mean()), int(((cov >= l) & (cov < h)).sum())]
                          for l, h in BINS}}
        report["policies"][key] = row
        print(f"  {key:8s} ref {row['ref']:.4f} cand {row['cand']:.4f} delta {row['delta']:+.4f} "
              f"CI95 [{lo:+.4f},{hi:+.4f}] median {row['median']:+.4f} W/L {row['wins']}/{row['losses']}")
        print("           by method " + " ".join(f"{u}:{v:+.4f}" for u, v in row["by_method"].items()))
        print("           by cov    " + " ".join(f"{b}:{v[0]:+.4f}(n={v[1]})" for b, v in row["by_cov"].items()))
    for name, recs in ((a.ref, r_ref), (a.cand, r_c)):
        print(f"  cost {name}: " + ", ".join(f"s{r['seed']} {r['t_total']:.0f}s {r['peak_mem_mb']:.0f}MB" for r in recs))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
