"""Two-pass clustering into family / model labels (TASK_PLAN_v2 steps 6, 8). No API calls.

Tune on the reference endpoints (known answers in practice_answers.csv):
    python cluster.py tune --ledger data/ref_ledger.jsonl
        -> tries feature weightings x thresholds, prints the best by practice MCC (tuned on non-held-out probes,
           validated on held-out probes), writes configs/tuned.json

Build a submission for the scored endpoints:
    python cluster.py solve --ledger data/scored_ledger.jsonl --config configs/tuned.json
        -> submissions/sub_<time>.csv + .json sidecar (config, hashes, clusters); validate and submit with submit.py

Check a config on the reference endpoints exactly as `solve` would use it:
    python cluster.py solve --target ref --ledger data/ref_ledger.jsonl --config configs/tuned.json

How distances are combined: each feature's pair distances are standardised (z-score over the pairs being clustered),
missing values set to the average, then a weighted mean is taken. Average-linkage clustering is cut at a threshold;
"auto" cuts at the largest gap between merge heights. Caveat: z-scores are relative to the set of endpoints, so a
threshold tuned on 9 single-family reference endpoints may not transfer 1:1 to 33 scored endpoints; check the cluster
sizes `solve` prints and the sidecar before submitting.
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
from features import FEATURES, compute, decode_mode
from validate_submission import validate


# ---------------------------------------------------------------- distance combination and cutting
def zscore(M):
    n = len(M)
    iu = np.triu_indices(n, 1)
    v = M[iu]
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return None
    sd = float(np.std(v[ok]))
    if sd == 0:
        return None
    Z = (M - float(np.mean(v[ok]))) / sd
    Z[~np.isfinite(Z)] = 0.0
    np.fill_diagonal(Z, 0.0)
    return Z


def combine(feats: dict, weights: dict, idx=None):
    acc, wsum = None, 0.0
    for f, w in weights.items():
        if not w or f not in feats:
            continue
        M = feats[f] if idx is None else feats[f][np.ix_(idx, idx)]
        Z = zscore(M)
        if Z is None:
            continue
        acc = w * Z if acc is None else acc + w * Z
        wsum += w
    n = len(feats[FEATURES[0]]) if idx is None else len(idx)
    if acc is None:
        return np.zeros((n, n))
    D = acc / wsum
    off = D[~np.eye(n, dtype=bool)]
    D = D - (off.min() if off.size else 0.0)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2


def cut(D, threshold):
    """Cluster ids (1..k) and the threshold used."""
    n = len(D)
    if n == 1:
        return np.array([1]), 0.0
    Z = linkage(squareform(D, checks=False), "average")
    h = Z[:, 2]
    if threshold == "auto":
        if len(h) == 1:
            t = h[0] / 2
        else:
            i = int(np.argmax(np.diff(h)))
            t = float((h[i] + h[i + 1]) / 2)
    else:
        t = float(threshold)
    return fcluster(Z, t, "distance"), t


def candidate_thresholds(D):
    if len(D) < 2:
        return [0.0]
    h = np.sort(linkage(squareform(D, checks=False), "average")[:, 2])
    mids = list((h[:-1] + h[1:]) / 2)
    return [float(h[0]) / 2] + mids + [float(h[-1]) + 1e-6]


# ---------------------------------------------------------------- scoring
def mcc_same_model(eps, pred: dict, truth: dict) -> float:
    """MCC over all pairs of `eps` for 'same (family, model)', as score_practice.py computes it."""
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


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


# ---------------------------------------------------------------- tune
def presets(available):
    yield "all_equal", {f: 1.0 for f in available}
    for f in available:
        yield f"only_{f}", {f: 1.0}
    if len(available) > 2:
        for f in available:
            yield f"all_but_{f}", {g: 1.0 for g in available if g != f}
    for a, b in itertools.combinations(available, 2):
        yield f"{a}+{b}", {a: 1.0, b: 1.0}


def cmd_tune(args):
    truth = read_labels(args.labels)
    eps, feats, profs = compute(args.ledger, args.probes, heldout="exclude")
    eps_l = [e for e in eps if e in truth]
    if len(eps_l) < 3:
        sys.exit(f"Need answers for at least 3 labelled endpoints in {args.ledger}; found {eps_l}.")
    idx = [eps.index(e) for e in eps_l]
    available = [f for f in FEATURES if zscore(feats[f][np.ix_(idx, idx)]) is not None]
    print(f"Endpoints: {', '.join(eps_l)}")
    print(f"Features with data: {', '.join(available)}")
    print("Decode modes: " + ", ".join(f"{e}={decode_mode(profs[e])}" for e in eps_l))

    _, feats_ho, _ = compute(args.ledger, args.probes, heldout="only", endpoints=eps)

    results = []
    for name, w in presets(available):
        D = combine(feats, w, idx)
        for t in candidate_thresholds(D) + ["auto"]:
            ids, t_used = cut(D, t)
            pred = {e: ids[k] for k, e in enumerate(eps_l)}
            results.append((mcc_same_model(eps_l, pred, truth), name, w, t, t_used))
    results.sort(key=lambda r: (-r[0], r[3] == "auto"))

    print(f"\nTop configurations (practice MCC on tuning probes; {len(results)} tried)")
    print(f"{'mcc':>6}  {'heldout':>7}  {'threshold':>9}  weights")
    best_rows = []
    for mcc, name, w, t, t_used in results[:args.top]:
        Dh = combine(feats_ho, w, idx)
        ids_h, _ = cut(Dh, t)
        mcc_h = mcc_same_model(eps_l, {e: ids_h[k] for k, e in enumerate(eps_l)}, truth)
        best_rows.append((mcc, mcc_h, name, w, t))
        print(f"{mcc:6.3f}  {mcc_h:7.3f}  {str(t if t == 'auto' else round(t_used, 3)):>9}  {name}")

    mcc, mcc_h, name, w, t = best_rows[0]
    cfg = json.loads(Path(args.base_config).read_text())
    cfg.update({"weights_identity": w, "identity_threshold": t,
                "tuned": {"preset": name, "practice_mcc": mcc, "heldout_mcc": mcc_h, "ledger": str(args.ledger),
                          "time": datetime.now().isoformat(timespec="seconds")}})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(f"\nBest: {name}, threshold={t}, practice MCC={mcc:.3f}, held-out MCC={mcc_h:.3f} -> {out}")
    print("36 pairs only: prefer a configuration that is good on BOTH columns over the single top line.")


# ---------------------------------------------------------------- solve
def cmd_solve(args):
    cfg = json.loads(Path(args.config).read_text())
    ref = args.target == "ref"
    want = [f"ref-{i:02d}" for i in range(1, 10)] if ref else config.scored_endpoints()
    eps, feats, profs = compute(args.ledger, args.probes, heldout=cfg.get("heldout", "all"), endpoints=want)
    empty = [e for e in eps if profs[e]["n"] == 0]
    if empty:
        print(f"WARNING: no answers for {', '.join(empty)}; they will be clustered on missing data.")

    # pass 1: families
    if ref:
        fam_ids = np.ones(len(eps), dtype=int)       # the reference endpoints are one family by construction
        t_fam = None
    else:
        D_fam = combine(feats, cfg["weights_family"])
        fam_ids, t_fam = cut(D_fam, cfg["family_threshold"])

    # pass 2: models inside each family
    rows, clusters = [], {}
    for fi, fam in enumerate(sorted(set(fam_ids))):
        idx = [k for k in range(len(eps)) if fam_ids[k] == fam]
        D = combine(feats, cfg["weights_identity"], idx)
        mod_ids, _ = cut(D, cfg["identity_threshold"]) if len(idx) > 1 else (np.array([1]), 0.0)
        fl = letters(fi)
        for k, m in zip(idx, mod_ids):
            rows.append((eps[k], fl, f"{fl}{m}"))
            clusters.setdefault(f"{fl}{m}", []).append(eps[k])

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
    side = {"csv": str(path), "config_file": args.config, "config_hash": cfg_hash, "config": cfg,
            "ledger": str(args.ledger), "ledger_sha256_12": ledger_hash, "family_threshold_used": t_fam,
            "clusters": clusters, "decode_modes": {e: decode_mode(profs[e]) for e in eps},
            "answers_per_endpoint": {e: profs[e]["n"] for e in eps}}
    path.with_suffix(".json").write_text(json.dumps(side, indent=2))

    fams = sorted(set(r[1] for r in rows))
    print(f"{len(fams)} families, {len(clusters)} models")
    for name, members in sorted(clusters.items()):
        print(f"  {name:>5}: {', '.join(members)}")
    errors = validate(path, practice=ref)
    print(f"\nWritten {path} (+ .json).  " + ("VALID" if not errors else "INVALID: " + "; ".join(errors)))
    if ref:
        truth = read_labels(args.labels)
        pred = {r[0]: (r[1], r[2]) for r in rows}
        print(f"Practice MCC: {mcc_same_model([e for e in eps if e in truth], pred, truth):.3f}"
              f"  (cross-check: python score_practice.py {path})")
    else:
        print(f"Next: python submit.py {path} --note \"...\"")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--ledger", default=str(config.REF_LEDGER))
    t.add_argument("--labels", default=str(config.PRACTICE_ANSWERS))
    t.add_argument("--probes", default=None)
    t.add_argument("--base-config", default=str(config.HERE / "configs" / "default.json"))
    t.add_argument("--out", default=str(config.HERE / "configs" / "tuned.json"))
    t.add_argument("--top", type=int, default=15)
    t.set_defaults(func=cmd_tune)

    s = sub.add_parser("solve")
    s.add_argument("--target", choices=["scored", "ref"], default="scored")
    s.add_argument("--ledger", default=None)
    s.add_argument("--config", default=str(config.HERE / "configs" / "default.json"))
    s.add_argument("--labels", default=str(config.PRACTICE_ANSWERS))
    s.add_argument("--probes", default=None)
    s.add_argument("--outdir", default=str(config.SUBMISSIONS_DIR))
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args(argv)
    if args.cmd == "solve" and args.ledger is None:
        args.ledger = str(config.REF_LEDGER if args.target == "ref" else config.SCORED_LEDGER)
    args.func(args)


if __name__ == "__main__":
    main()
