"""Parse answers and compute calibrated pairwise evidence (TASK_PLAN_v2 steps 4-5). No API calls.

    python features.py data/ref_ledger.jsonl                         # diagnostics + pair table
    python features.py data/ref_ledger.jsonl --repeats 0:3           # only repeats 0,1,2 (match the scored budget)
    python features.py data/scored_ledger.jsonl --out data/scored_pairs.csv

For every pair of endpoints and every feature, the observed distance is compared with a PERMUTATION NULL: the answers
of both endpoints to the same probe are pooled and randomly re-split many times. The feature value is the z-score
    z = (observed - mean(permuted)) / std(permuted)
so z ~ 0 when the two endpoints are indistinguishable with the data at hand and z >> 0 when they clearly differ.
Unlike a ranking relative to the other endpoints, this scale does not depend on which endpoints are clustered together,
so a threshold tuned on the reference endpoints transfers to the scored ones as long as the number of answers per
probe is similar (use --repeats when tuning to match the scored budget).

Features (each summed over the probes of its type before the z-score):
    choice   Jensen-Shannon divergence of answer distributions on `choice` probes (failed parses dropped)
    bits     L1 distance of bit statistics (P(1), transitions, run lengths, length) on `bits` probes
    numbers  L1 distance of number histograms (+ adjacent-repeat rate) on `numbers` probes
    prefix   1 - mean common-prefix ratio across endpoints on `greedy_long` probes
    mmd      unbiased MMD^2, character Hamming kernel on the first 64 chars, on every probe
"""
import argparse
import csv
import itertools
import os
import re
from collections import defaultdict

import numpy as np

import config
from ledger import load_records

FEATURES = ["choice", "bits", "numbers", "prefix", "mmd"]
NATURAL = {"choice": "choice", "bits": "bits", "numbers": "numbers", "greedy_long": "prefix", "decode": "mmd"}
WORD = re.compile(r"[a-z0-9]+")
MMD_LEN = 64
MAX_RUN = 8
MIN_PER_SIDE = 2          # answers per endpoint and probe needed to use that probe for a pair


# ---------------------------------------------------------------- loading and parsing
def parse_repeats(spec):
    """'0:3' -> (0, 3); None -> None."""
    if not spec:
        return None
    lo, hi = spec.split(":")
    return int(lo), int(hi)


def load_answers(ledger_path, probes: dict, heldout: str = "all", repeats=None, probe_ids=None) -> dict:
    """{endpoint: {probe_id: [response, ...]}}. heldout: all | exclude | only. repeats: (lo, hi) or None.
    probe_ids: only these probes (None = all)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in load_records(ledger_path):
        if r.get("status") != "ok" or r.get("response") is None:
            continue
        p = probes.get(r["probe_id"])
        if p is None or (probe_ids is not None and r["probe_id"] not in probe_ids):
            continue
        is_ho = bool(p.get("heldout"))
        if (heldout == "exclude" and is_ho) or (heldout == "only" and not is_ho):
            continue
        if repeats and not repeats[0] <= int(r.get("repeat", 0)) < repeats[1]:
            continue
        out[r["endpoint"]][r["probe_id"]].append(r["response"])
    return out


def parse_choice(text: str, options=None):
    """Option word found in the answer (first one), or None when the answer does not follow the format."""
    words = WORD.findall(text.lower())
    if options:
        opts = {o.lower() for o in options}
        return next((w for w in words if w in opts), None)
    return words[0] if words else None


def parse_bits(text: str, alphabet: str = "01") -> str:
    zero, one = alphabet[0].upper(), alphabet[1].upper()
    return "".join("0" if c == zero else "1" for c in text.upper() if c in (zero, one))


def parse_numbers(text: str, lo: int, hi: int) -> list:
    return [v for v in (int(x) for x in re.findall(r"-?\d+", text)) if lo <= v <= hi]


def bits_row(s: str) -> np.ndarray:
    """Sufficient statistics of one bit string: [1, n, n1, t00, t01, t10, t11, runs(1..MAX_RUN+)]."""
    t = np.zeros(4)
    for a, b in zip(s, s[1:]):
        t[2 * int(a) + int(b)] += 1
    runs = np.zeros(MAX_RUN)
    for m in re.finditer(r"0+|1+", s):
        runs[min(len(m.group()), MAX_RUN) - 1] += 1
    return np.concatenate([[1, len(s), s.count("1")], t, runs])


def numbers_row(vals: list, lo: int, hi: int) -> np.ndarray:
    hist = np.bincount(np.array(vals) - lo, minlength=hi - lo + 1).astype(float)
    pairs = max(len(vals) - 1, 0)
    reps = sum(a == b for a, b in zip(vals, vals[1:]))
    return np.concatenate([hist, [pairs, reps]])


def encode(text: str) -> np.ndarray:
    x = np.zeros(MMD_LEN, dtype=np.int64)
    codes = [ord(c) for c in text[:MMD_LEN]]
    x[:len(codes)] = codes
    return x


def profile(answers_ep: dict, probes: dict) -> dict:
    """Parsed answers per probe, plus diagnostics (decode duplicates, parse failures)."""
    prof = {"probes": {}, "n": 0, "decode_dup": None, "fail": defaultdict(lambda: [0, 0])}
    dups = []
    for pid, resps in answers_ep.items():
        p = probes[pid]
        t = p["type"]
        prof["n"] += len(resps)
        e = {"type": t, "raw": resps, "enc": np.array([encode(r) for r in resps])}
        if t == "decode" and len(resps) >= 2:
            pairs = list(itertools.combinations(resps, 2))
            dups.append(sum(a == b for a, b in pairs) / len(pairs))
        elif t == "choice":
            labels = [parse_choice(r, p.get("options")) for r in resps]
            e["labels"] = [x for x in labels if x is not None]
            prof["fail"][t][0] += labels.count(None)
        elif t == "bits":
            seqs = [parse_bits(r, p.get("alphabet", "01")) for r in resps]
            ok = [s for s in seqs if len(s) >= 10]
            e["S"] = np.array([bits_row(s) for s in ok]) if ok else None
            prof["fail"][t][0] += len(seqs) - len(ok)
        elif t == "numbers":
            lists = [parse_numbers(r, p["lo"], p["hi"]) for r in resps]
            ok = [v for v in lists if len(v) >= 5]
            e["S"] = np.array([numbers_row(v, p["lo"], p["hi"]) for v in ok]) if ok else None
            prof["fail"][t][0] += len(lists) - len(ok)
        prof["fail"][t][1] += len(resps)
        prof["probes"][pid] = e
    if dups:
        prof["decode_dup"] = float(np.mean(dups))
    return prof


def decode_mode(prof) -> str:
    d = prof["decode_dup"]
    if d is None:
        return "unknown"
    return "greedy" if d >= 0.999 else "sampling"


# ---------------------------------------------------------------- statistics on summed sufficient statistics
def _js_rows(A, B):
    pa = A + 0.5
    pb = B + 0.5
    pa = pa / pa.sum(-1, keepdims=True)
    pb = pb / pb.sum(-1, keepdims=True)
    m = (pa + pb) / 2
    return 0.5 * (pa * np.log2(pa / m)).sum(-1) + 0.5 * (pb * np.log2(pb / m)).sum(-1)


def _bits_f(T):
    cnt, n, n1, t = T[:, 0], T[:, 1], T[:, 2], T[:, 3:7]
    runs = T[:, 7:]
    p1 = n1 / np.maximum(n, 1)
    p10 = t[:, 1] / np.maximum(t[:, 0] + t[:, 1], 1)
    p11 = t[:, 3] / np.maximum(t[:, 2] + t[:, 3], 1)
    mean_len = np.minimum(n / np.maximum(cnt, 1) / 300.0, 1.0)
    runs = runs / np.maximum(runs.sum(-1, keepdims=True), 1)
    return np.column_stack([p1, p10, p11, mean_len, runs])


def _numbers_f(T):
    hist = T[:, :-2]
    hist = hist / np.maximum(hist.sum(-1, keepdims=True), 1)
    rep = T[:, -1] / np.maximum(T[:, -2], 1)
    return np.column_stack([hist, rep])


def _l1(f):
    return lambda A, B: 0.5 * np.abs(f(A) - f(B)).sum(-1)


SUM_STATS = {"bits": _l1(_bits_f), "numbers": _l1(_numbers_f), "choice": _js_rows}


def _perm_split(rng, n_a, n_b, P):
    idx = np.argsort(rng.random((P, n_a + n_b)), axis=1)
    return idx[:, :n_a], idx[:, n_a:]


def sum_test(Sa, Sb, stat, rng, P):
    """(observed, permuted[P]) for a statistic of the summed rows of each side."""
    S = np.vstack([Sa, Sb])
    obs = stat(Sa.sum(0, keepdims=True), Sb.sum(0, keepdims=True))[0]
    ia, ib = _perm_split(rng, len(Sa), len(Sb), P)
    return float(obs), stat(S[ia].sum(1), S[ib].sum(1))


def _mmd_idx(K, ia, ib):
    na, nb = ia.shape[1], ib.shape[1]
    saa = (K[ia[:, :, None], ia[:, None, :]].sum((1, 2)) - na) / (na * (na - 1))
    sbb = (K[ib[:, :, None], ib[:, None, :]].sum((1, 2)) - nb) / (nb * (nb - 1))
    sab = K[ia[:, :, None], ib[:, None, :]].mean((1, 2))
    return saa + sbb - 2 * sab


def _prefix_idx(K, ia, ib):
    return 1.0 - K[ia[:, :, None], ib[:, None, :]].mean((1, 2))


def kernel_test(K, n_a, n_b, fn, rng, P):
    obs = fn(K, np.arange(n_a)[None], np.arange(n_a, n_a + n_b)[None])[0]
    ia, ib = _perm_split(rng, n_a, n_b, P)
    return float(obs), fn(K, ia, ib)


def hamming_K(Xa, Xb):
    X = np.vstack([Xa, Xb])
    return (X[:, None, :] == X[None, :, :]).mean(-1)


def prefix_K(ra, rb):
    R = list(ra) + list(rb)
    n = len(R)
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = min(len(R[i]), len(R[j]))
            K[i, j] = K[j, i] = len(os.path.commonprefix([R[i], R[j]])) / m if m else 0.0
    return K


def pair_tests(ea: dict, eb: dict, rng, P: int):
    """{probe_id: {feature: (observed, permuted)}} for the probes both endpoints answered often enough."""
    out = {}
    for pid in sorted(ea.keys() & eb.keys()):          # sorted: same random draws on every run (reproducible)
        a, b = ea[pid], eb[pid]
        t = a["type"]
        res = {}
        if len(a["raw"]) >= MIN_PER_SIDE and len(b["raw"]) >= MIN_PER_SIDE:
            res["mmd"] = kernel_test(hamming_K(a["enc"], b["enc"]), len(a["raw"]), len(b["raw"]), _mmd_idx, rng, P)
        if t == "choice" and len(a["labels"]) >= MIN_PER_SIDE and len(b["labels"]) >= MIN_PER_SIDE:
            cats = sorted(set(a["labels"]) | set(b["labels"]))
            pos = {c: i for i, c in enumerate(cats)}
            Sa = np.zeros((len(a["labels"]), len(cats)))
            Sb = np.zeros((len(b["labels"]), len(cats)))
            for i, x in enumerate(a["labels"]):
                Sa[i, pos[x]] = 1
            for i, x in enumerate(b["labels"]):
                Sb[i, pos[x]] = 1
            res["choice"] = sum_test(Sa, Sb, SUM_STATS["choice"], rng, P)
        elif t in ("bits", "numbers") and a["S"] is not None and b["S"] is not None \
                and len(a["S"]) >= MIN_PER_SIDE and len(b["S"]) >= MIN_PER_SIDE:
            res[t] = sum_test(a["S"], b["S"], SUM_STATS[t], rng, P)
        elif t == "greedy_long" and len(a["raw"]) >= MIN_PER_SIDE and len(b["raw"]) >= MIN_PER_SIDE:
            res["prefix"] = kernel_test(prefix_K(a["raw"], b["raw"]), len(a["raw"]), len(b["raw"]), _prefix_idx,
                                        rng, P)
        if res:
            out[pid] = res
    return out


def _z(obs, perm):
    sd = float(np.std(perm))
    return (obs - float(np.mean(perm))) / max(sd, 1e-6 + 0.05 * abs(float(np.mean(perm))))


def compute(ledger_path, probes_path=None, heldout="all", endpoints=None, repeats=None, perms=200, seed=0,
            probe_ids=None):
    """Returns dict with
        eps      endpoint list
        z        {feature: n x n z-score matrix, NaN where no data}
        raw      {feature: n x n observed distance}
        probe_z  {probe_id: n x n z of that probe's natural feature}
        profs    {endpoint: profile}"""
    probes = config.load_probes(probes_path)
    answers = load_answers(ledger_path, probes, heldout, repeats, probe_ids)
    eps = sorted(endpoints if endpoints is not None else answers.keys())
    profs = {e: profile(answers.get(e, {}), probes) for e in eps}
    n = len(eps)
    z = {f: np.full((n, n), np.nan) for f in FEATURES}
    raw = {f: np.full((n, n), np.nan) for f in FEATURES}
    probe_z = defaultdict(lambda: np.full((n, n), np.nan))
    rng = np.random.default_rng(seed)
    for i, j in itertools.combinations(range(n), 2):
        tests = pair_tests(profs[eps[i]]["probes"], profs[eps[j]]["probes"], rng, perms)
        for f in FEATURES:
            parts = [res[f] for res in tests.values() if f in res]
            if parts:
                obs = sum(o for o, _ in parts)
                perm = np.sum([p for _, p in parts], axis=0)
                z[f][i, j] = z[f][j, i] = _z(obs, perm)
                raw[f][i, j] = raw[f][j, i] = obs
        for pid, res in tests.items():
            nat = NATURAL.get(probes[pid]["type"])
            if nat in res:
                probe_z[pid][i, j] = probe_z[pid][j, i] = _z(*res[nat])
    return {"eps": eps, "z": z, "raw": raw, "probe_z": dict(probe_z), "profs": profs, "probes": probes}


# ---------------------------------------------------------------- report
def print_diagnostics(res) -> None:
    eps, profs = res["eps"], res["profs"]
    types = ["choice", "bits", "numbers"]
    print(f"{'endpoint':>9} {'answers':>7} {'decode':>9} {'dup':>5}  " + "  ".join(f"fail_{t:<7}" for t in types))
    for e in eps:
        p = profs[e]
        dup = "" if p["decode_dup"] is None else f"{p['decode_dup']:.2f}"
        fails = []
        for t in types:
            bad, tot = p["fail"][t] if t in p["fail"] else (0, 0)
            fails.append(f"{(bad / tot if tot else 0):>11.0%}" if tot else f"{'-':>11}")
        print(f"{e:>9} {p['n']:>7} {decode_mode(p):>9} {dup:>5}  " + "  ".join(fails))
    worst = [(e, t, p["fail"][t]) for e in eps for t in types if t in (p := profs[e])["fail"]
             and p["fail"][t][1] and p["fail"][t][0] / p["fail"][t][1] > 0.3]
    if worst:
        print("\nParse failures above 30% (answers ignored; reword the probe if this is common):")
        for e, t, (bad, tot) in worst:
            sample = next((pr["raw"][0] for pr in profs[e]["probes"].values() if pr["type"] == t), "")
            print(f"  {e} {t}: {bad}/{tot}   e.g. {sample[:100]!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--heldout", choices=["all", "exclude", "only"], default="all")
    ap.add_argument("--repeats", default=None, help="lo:hi, only use these repeat indices (e.g. 0:3)")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", default=None, help="write the pair table (z and raw) as CSV")
    args = ap.parse_args(argv)

    res = compute(args.ledger, args.probes, args.heldout, repeats=parse_repeats(args.repeats), perms=args.perms)
    eps = res["eps"]
    if not eps:
        raise SystemExit(f"No answered requests in {args.ledger}.")
    print_diagnostics(res)

    pairs = list(itertools.combinations(range(len(eps)), 2))
    print("\nz-scores (0 = indistinguishable, >3 = clearly different)")
    print(f"{'a':>7} {'b':>7} " + " ".join(f"{f:>8}" for f in FEATURES))
    for i, j in pairs[:60]:
        print(f"{eps[i]:>7} {eps[j]:>7} " + " ".join(f"{res['z'][f][i, j]:8.2f}" for f in FEATURES))
    if len(pairs) > 60:
        print(f"... {len(pairs) - 60} more pairs (use --out)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b"] + [f"z_{x}" for x in FEATURES] + [f"raw_{x}" for x in FEATURES])
            for i, j in pairs:
                w.writerow([eps[i], eps[j]] + [res["z"][x][i, j] for x in FEATURES]
                           + [res["raw"][x][i, j] for x in FEATURES])
        print(f"Pair table written to {args.out}")


if __name__ == "__main__":
    main()
