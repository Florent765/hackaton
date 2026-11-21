"""Two-pass clustering into family / model labels (TASK_PLAN_v2 steps 6, 8). No API calls.

Tune on the reference endpoints (known answers in practice_answers.csv, unlimited queries):
    python cluster.py tune
        1. ranks every probe by how well it separates different models from disguised copies of one model (AUC);
        2. allocation search: for K = 1, 2, 3, ... best probes with budget // K answers each (default budget 90, which
           leaves 10 answers per scored endpoint in reserve), measures the practice MCC by leave-one-block-out
           cross-validation over disjoint blocks of reference repeats (each block holds exactly the data a scored
           endpoint will have), for every feature weighting;
        3. writes configs/tuned.json: probes to query, answers per probe, weights, identity threshold. If some K cannot
           be tested yet, it prints the reference collection command to run.

Family pass (default): families are cut on the same answers as the models, with weights_family and threshold "auto"
(largest gap between merges); models are then cut inside each family.
Optional, OFF by default: dedicated family probes. List them in "family_probes" in configs/default.json and run tune
with --budget 80. tune then calibrates the family threshold above the largest family distance the (single-family)
references show and prints a second scored collection command. It measured worse on the local mock (see README).

Build a submission for the scored endpoints:
    python cluster.py solve --config configs/tuned.json
        -> submissions/sub_<time>.csv + .json sidecar; prints the clusters and the pairs closest to the threshold
           (where the reserve budget should go). Validate and submit with submit.py.

Check a config on the reference endpoints exactly as `solve` would use it:
    python cluster.py solve --target ref --config configs/tuned.json --repeats 0:R
        (R = repeats_per_probe written in tuned.json, so the check uses as many answers as a scored endpoint)

Distance between two endpoints = weighted mean of the per-feature permutation z-scores (negative z set to 0, missing
features ignored). Average-linkage clustering is cut at an absolute threshold on that scale. The scale is calibrated
per pair (see features.py), so a threshold tuned on the references transfers to the scored endpoints when the number of
answers per probe is the same; that is why tuning works on blocks of exactly budget // K reference repeats.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import config
from features import FEATURES, compute, decode_mode, parse_repeats
from validate_submission import validate

THRESHOLDS = np.round(np.concatenate([np.arange(0.25, 10, 0.25), np.arange(10, 40, 1.0)]), 2)


# ---------------------------------------------------------------- distances and cutting
def combine(z: dict, weights: dict, idx=None):
    acc = wsum = None
    for f, w in weights.items():
        if not w or f not in z:
            continue
        M = z[f] if idx is None else z[f][np.ix_(idx, idx)]
        has = np.isfinite(M)
        Mc = np.where(has, np.clip(M, 0, None), 0.0)
        acc = w * Mc if acc is None else acc + w * Mc
        wsum = w * has if wsum is None else wsum + w * has
    n = len(next(iter(z.values()))) if idx is None else len(idx)
    if acc is None:
        return np.zeros((n, n))
    D = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0.0)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2


def cut(D, threshold):
    """Cluster ids (1..k) and the threshold used. threshold: number, or 'auto' (largest gap between merges)."""
    n = len(D)
    if n == 1:
        return np.array([1]), 0.0
    Z = linkage(squareform(D, checks=False), "average")
    if threshold == "auto":
        h = Z[:, 2]
        if len(h) == 1:
            t = h[0] / 2
        else:
            i = int(np.argmax(np.diff(h)))
            t = float((h[i] + h[i + 1]) / 2)
    else:
        t = float(threshold)
    return fcluster(Z, t, "distance"), t


# ---------------------------------------------------------------- scoring helpers
def mcc_same_model(eps, pred: dict, truth: dict) -> float:
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def auc(pos, neg) -> float:
    """P(random positive > random negative), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def read_labels(path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["endpoint"].strip()] = (r["family"].strip(), r["model"].strip())
    return out


def letters(i: int) -> str:
    name, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def presets(available):
    yield "all_equal", {f: 1.0 for f in available}
    for f in available:
        yield f"only_{f}", {f: 1.0}
    if len(available) > 2:
        for f in available:
            yield f"all_but_{f}", {g: 1.0 for g in available if g != f}
    for a, b in itertools.combinations(available, 2):
        yield f"{a}+{b}", {a: 1.0, b: 1.0}


def widest_optimum(scores):
    """Index of the middle of the longest run of thresholds within 0.01 of the best score, and that run's length."""
    best = max(scores)
    good = [s >= best - 0.01 for s in scores]
    best_run, run_start = (0, 0), None
    for k, g in enumerate(good + [False]):
        if g and run_start is None:
            run_start = k
        elif not g and run_start is not None:
            if k - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, k)
            run_start = None
    lo, hi = best_run
    return (lo + hi - 1) // 2, hi - lo


# ---------------------------------------------------------------- tune
def available_repeats(ledger_path, eps) -> dict:
    """{probe_id: number of repeats answered on EVERY given reference endpoint}."""
    from ledger import load_records
    have = {}
    for r in load_records(ledger_path):
        if r.get("status") == "ok" and r["endpoint"] in eps:
            have.setdefault(r["probe_id"], {}).setdefault(r["endpoint"], set()).add(int(r.get("repeat", 0)))
    return {pid: min(len(per.get(e, ())) for e in eps) for pid, per in have.items()}


def block_mcc(Ds, t, eps, truth):
    return [mcc_same_model(eps, {e: ids[k] for k, e in enumerate(eps)}, truth) for ids in (cut(D, t)[0] for D in Ds)]


def cv_score(Ds, eps, truth):
    """Leave-one-block-out: pick the threshold on the other blocks, score the held-out block. Returns
    (cv_mcc, threshold chosen on all blocks, width of its optimal range, mean MCC at that threshold)."""
    table = np.array([block_mcc(Ds, t, eps, truth) for t in THRESHOLDS])      # thresholds x blocks
    cv = []
    for b in range(len(Ds)):
        others = np.delete(table, b, axis=1).mean(1)
        k, _ = widest_optimum(list(others))
        cv.append(table[k, b])
    k, width = widest_optimum(list(table.mean(1)))
    return float(np.mean(cv)), float(THRESHOLDS[k]), width, float(table[k].mean())


def cmd_tune(args):
    truth = read_labels(args.labels)
    probes = config.load_probes(args.probes)
    ref_eps = list(truth)
    avail = available_repeats(args.ledger, ref_eps)
    if not avail:
        sys.exit(f"No reference answers in {args.ledger}. Run: python query.py run --target ref --repeats 6")

    # 1. probe ranking with a few answers per probe, averaged over disjoint blocks
    rr = args.rank_repeats
    n_rank_blocks = max(1, min(3, min(avail.values()) // rr))
    rank_res = [compute(args.ledger, args.probes, endpoints=ref_eps, repeats=(b * rr, (b + 1) * rr),
                        perms=args.perms, seed=b) for b in range(n_rank_blocks)]
    eps = [e for e in rank_res[0]["eps"] if e in truth]
    idx = [rank_res[0]["eps"].index(e) for e in eps]
    print(f"Reference endpoints: {', '.join(eps)}")
    print("Decode modes: " + ", ".join(f"{e}={decode_mode(rank_res[0]['profs'][e])}" for e in eps))
    same = [(a, b) for a, b in itertools.combinations(range(len(eps)), 2) if truth[eps[a]] == truth[eps[b]]]
    diff = [(a, b) for a, b in itertools.combinations(range(len(eps)), 2) if truth[eps[a]] != truth[eps[b]]]

    print(f"\n1. Probe ranking ({rr} answers per probe, {n_rank_blocks} blocks). AUC 1.0 = separates different "
          f"models from same-model disguises every time, 0.5 = useless")
    print(f"{'probe':>14} {'type':>11} {'AUC':>6} {'z_same':>7} {'z_diff':>7} {'ref_repeats':>11}")
    ranking = []
    for pid, p in probes.items():
        mats = [r["probe_z"][pid][np.ix_(idx, idx)] for r in rank_res if pid in r["probe_z"]]
        zs = np.array([m[a, b] for m in mats for a, b in same])
        zd = np.array([m[a, b] for m in mats for a, b in diff])
        zs, zd = zs[np.isfinite(zs)], zd[np.isfinite(zd)]
        if len(zs) and len(zd):
            ranking.append((auc(zd, zs), pid, p["type"], float(np.median(zs)), float(np.median(zd))))
    ranking.sort(reverse=True)
    for a, pid, t, ms, md in ranking:
        print(f"{pid:>14} {t:>11} {a:6.2f} {ms:7.2f} {md:7.2f} {avail.get(pid, 0):>11}")
    order = [pid for a, pid, *_ in ranking]

    # 2. allocation search: K best probes x (budget // K) answers each, cross-validated over blocks of references
    print(f"\n2. Allocation search for a budget of {args.budget} answers per scored endpoint "
          f"(leave-one-block-out MCC on the references)")
    results, missing = [], []
    for K in [int(k) for k in args.ks.split(",")]:
        if K > len(order):
            continue
        chosen = order[:K]
        R = args.budget // K
        if R < 2:
            continue
        nb = min(avail.get(pid, 0) for pid in chosen) // R
        if nb < 2:
            missing.append((K, R, chosen))
            continue
        nb = min(nb, args.max_blocks)
        res = [compute(args.ledger, args.probes, endpoints=ref_eps, repeats=(b * R, (b + 1) * R), perms=args.perms,
                       seed=100 + b, probe_ids=set(chosen)) for b in range(nb)]
        available = [f for f in FEATURES if all(np.isfinite(r["z"][f][np.ix_(idx, idx)]).any() for r in res)]
        for name, w in presets(available):
            Ds = [combine(r["z"], w, idx) for r in res]
            cv, t, width, full = cv_score(Ds, eps, truth)
            results.append((cv, width, full, K, R, nb, name, w, t, chosen))
    for K, R, chosen in missing:
        need = 2 * R
        print(f"  K={K}: needs {need} reference repeats of {', '.join(chosen)} (to test {R} answers each). Collect:\n"
              f"      python query.py run --target ref --ids {','.join(chosen)} --repeats {need}")
    if not results:
        sys.exit("No allocation could be tested yet: collect the reference repeats listed above, then re-run tune.")
    results.sort(key=lambda r: (-round(r[0], 3), -r[1], -round(r[2], 3)))
    print(f"\n{'cv_mcc':>6} {'full':>5} {'width':>5} {'K':>2} {'R':>3} {'blocks':>6} {'thresh':>6}  weights / probes")
    for cv, width, full, K, R, nb, name, w, t, chosen in results[:args.top]:
        print(f"{cv:6.3f} {full:5.2f} {width:>5} {K:>2} {R:>3} {nb:>6} {t:6.2f}  {name} / {','.join(chosen)}")

    cv, width, full, K, R, nb, name, w, t, chosen = results[0]
    cfg = json.loads(Path(args.base_config).read_text())

    # 3. OPTIONAL dedicated family probes (only when "family_probes" is set in the base config). The references are a
    # single family, so the largest family distance they show (disjoint blocks of family_repeats answers) is what
    # disguises and fine-tunes of ONE base model look like; the family threshold is set family_margin x above it.
    fam_probes = [pid for pid in cfg.get("family_probes", []) if pid in probes]
    Rf = args.family_repeats
    nbf = min([avail.get(pid, 0) for pid in fam_probes] or [0]) // Rf
    t_fam, fam_spread = cfg.get("family_threshold", "auto"), None
    if fam_probes and nbf < 1:
        print(f"\n3. Family probes need {Rf} reference repeats each; collect them with\n"
              f"      python query.py run --target ref --ids {','.join(fam_probes)} --repeats {2 * Rf}\n"
              f"   and re-run tune. Using the default family pass for now.")
        fam_probes = []
    if fam_probes:
        heights = []
        for b in range(min(nbf, 4)):
            r = compute(args.ledger, args.probes, endpoints=ref_eps, repeats=(b * Rf, (b + 1) * Rf), perms=args.perms,
                        seed=500 + b, probe_ids=set(fam_probes))
            Df = combine(r["z"], cfg["weights_family"], idx)
            heights.append(float(linkage(squareform(Df, checks=False), "average")[:, 2].max()))
        fam_spread = max(heights)
        t_fam = round(max(args.family_margin * fam_spread, t, 1.0), 2)
        print(f"\n3. Family pass: probes {', '.join(fam_probes)} x {Rf} answers; largest within-family merge on the "
              f"references {fam_spread:.2f} over {len(heights)} blocks -> family threshold {t_fam}")
    else:
        print(f"\n3. Family pass: default (same answers as the models, threshold {t_fam!r}). Dedicated family probes "
              f"are off; set family_probes in the base config to try them.")

    cfg.update({"weights_identity": w, "identity_threshold": t, "recommended_probes": chosen,
                "repeats_per_probe": R, "family_probes": fam_probes, "family_repeats": Rf,
                "family_threshold": t_fam, "family_spread_on_refs": fam_spread,
                "tuned": {"preset": name, "cv_mcc": cv, "mcc_at_threshold": full, "optimal_range_width": int(width),
                          "budget": args.budget, "blocks": nb, "ledger": str(args.ledger),
                          "time": datetime.now().isoformat(timespec="seconds")}})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(f"\nBest: {K} probes x {R} answers ({', '.join(chosen)}), {name}, threshold {t}, "
          f"cross-validated MCC {cv:.3f} -> {out}")
    total = K * R + len(fam_probes) * Rf
    print(f"Scored collection for this plan ({total} answers per scored endpoint; run it without --yes first to "
          f"read the plan):\n    python query.py run --target scored --recommended {out} --repeats {R} "
          f"--max-total {K * R} --yes")
    if fam_probes:
        print(f"    python query.py run --target scored --ids {','.join(fam_probes)} --repeats {Rf} "
              f"--max-total {total} --yes")
    if cv < 0.8:
        print("WARNING: cross-validated MCC below 0.8 on the references. Before spending scored budget: collect "
              "more reference repeats for the K values listed above, add or reword probes, and re-tune.")


# ---------------------------------------------------------------- solve
def cmd_solve(args):
    cfg = json.loads(Path(args.config).read_text())
    ref = args.target == "ref"
    want = [f"ref-{i:02d}" for i in range(1, 10)] if ref else config.scored_endpoints()
    rec = cfg.get("recommended_probes") or None
    res = compute(args.ledger, args.probes, heldout=cfg.get("heldout", "all"), endpoints=want,
                  repeats=parse_repeats(args.repeats), perms=args.perms, probe_ids=set(rec) if rec else None)
    eps, z, profs = res["eps"], res["z"], res["profs"]
    empty = [e for e in eps if profs[e]["n"] == 0]
    if empty:
        print(f"WARNING: no answers for {', '.join(empty)}.")

    if ref:
        fam_ids, t_fam = np.ones(len(eps), dtype=int), None   # the references are one family by construction
    else:
        fp = cfg.get("family_probes") or []
        if fp:      # optional dedicated family probes (see tune)
            zf = compute(args.ledger, args.probes, endpoints=want, perms=args.perms, seed=7, probe_ids=set(fp))["z"]
        else:       # default: families on the same answers as the models
            zf = z
        fam_ids, t_fam = cut(combine(zf, cfg["weights_family"]), cfg["family_threshold"])

    t_id = cfg["identity_threshold"]
    t_id_used = float(t_id) if t_id != "auto" else 0.0
    rows, clusters, near = [], {}, []
    for fi, fam in enumerate(sorted(set(fam_ids))):
        idx = [k for k in range(len(eps)) if fam_ids[k] == fam]
        D = combine(z, cfg["weights_identity"], idx)
        mod_ids = cut(D, t_id)[0] if len(idx) > 1 else np.array([1])
        fl = letters(fi)
        for k, m in zip(idx, mod_ids):
            rows.append((eps[k], fl, f"{fl}{m}"))
            clusters.setdefault(f"{fl}{m}", []).append(eps[k])
        if t_id != "auto":
            for a, b in itertools.combinations(range(len(idx)), 2):
                if abs(D[a, b] - t_id_used) <= args.margin:
                    near.append((round(float(D[a, b]), 2), eps[idx[a]], eps[idx[b]]))

    rows.sort()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{'practice' if ref else 'sub'}_{stamp}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)

    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    ledger_hash = hashlib.sha256(Path(args.ledger).read_bytes()).hexdigest()[:12]
    near.sort(key=lambda r: abs(r[0] - t_id_used))
    side = {"csv": str(path), "config_file": args.config, "config_hash": cfg_hash, "config": cfg,
            "ledger": str(args.ledger), "ledger_sha256_12": ledger_hash, "family_threshold_used": t_fam,
            "clusters": clusters, "near_threshold_pairs": near,
            "decode_modes": {e: decode_mode(profs[e]) for e in eps},
            "answers_per_endpoint": {e: profs[e]["n"] for e in eps}}
    path.with_suffix(".json").write_text(json.dumps(side, indent=2))

    print(f"{len(set(r[1] for r in rows))} families, {len(clusters)} models")
    for name, members in sorted(clusters.items()):
        print(f"  {name:>5}: {', '.join(members)}")
    if near:
        print(f"\nPairs within {args.margin} of the identity threshold {t_id_used:.2f} (spend the reserve here):")
        for d, a, b in near[:15]:
            print(f"  {a} {b}  distance {d}")
    errors = validate(path, practice=ref)
    print(f"\nWritten {path} (+ .json).  " + ("VALID" if not errors else "INVALID: " + "; ".join(errors)))
    if ref:
        truth = read_labels(args.labels)
        pred = {r[0]: (r[1], r[2]) for r in rows}
        print(f"Practice MCC: {mcc_same_model([e for e in eps if e in truth], pred, truth):.3f}"
              f"  (cross-check: python score_practice.py {path})")
    else:
        sizes = sorted((len(m) for m in clusters.values()), reverse=True)
        print(f"Model cluster sizes: {sizes}")
        if len(clusters) in (1, len(eps)):
            print("WARNING: everything merged or everything split. Check the threshold before submitting.")
        print(f"Next: python submit.py {path} --note \"...\"")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--ledger", default=str(config.REF_LEDGER))
    t.add_argument("--labels", default=str(config.PRACTICE_ANSWERS))
    t.add_argument("--probes", default=None)
    t.add_argument("--budget", type=int, default=90,
                   help="scored answers per endpoint for the identity probes (90 keeps 10 in reserve; 80 with "
                        "family probes)")
    t.add_argument("--family-repeats", type=int, default=2, help="answers per family probe (only with family_probes)")
    t.add_argument("--family-margin", type=float, default=1.5,
                   help="family threshold = margin x largest family distance on the references (only with "
                        "family_probes)")
    t.add_argument("--ks", default="1,2,3,4,6,9", help="numbers of probes to try")
    t.add_argument("--rank-repeats", type=int, default=3, help="answers per probe in the ranking pass")
    t.add_argument("--max-blocks", type=int, default=4, help="max reference blocks per allocation")
    t.add_argument("--perms", type=int, default=200)
    t.add_argument("--base-config", default=str(config.HERE / "configs" / "default.json"))
    t.add_argument("--out", default=str(config.HERE / "configs" / "tuned.json"))
    t.add_argument("--top", type=int, default=12)
    t.set_defaults(func=cmd_tune)

    s = sub.add_parser("solve")
    s.add_argument("--target", choices=["scored", "ref"], default="scored")
    s.add_argument("--ledger", default=None)
    s.add_argument("--config", default=str(config.HERE / "configs" / "default.json"))
    s.add_argument("--labels", default=str(config.PRACTICE_ANSWERS))
    s.add_argument("--probes", default=None)
    s.add_argument("--repeats", default=None, help="lo:hi, only these repeats (for ref checks)")
    s.add_argument("--perms", type=int, default=200)
    s.add_argument("--margin", type=float, default=1.5, help="report pairs this close to the threshold")
    s.add_argument("--outdir", default=str(config.SUBMISSIONS_DIR))
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args(argv)
    if args.cmd == "solve" and args.ledger is None:
        args.ledger = str(config.REF_LEDGER if args.target == "ref" else config.SCORED_LEDGER)
    args.func(args)


if __name__ == "__main__":
    main()
