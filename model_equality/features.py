"""Parse answers and compute pairwise endpoint distances (TASK_PLAN_v2 steps 4-5). No API calls.

    python features.py data/ref_ledger.jsonl                      # profile per endpoint + pair table on screen
    python features.py data/ref_ledger.jsonl --out data/ref_pairs.csv
    python features.py data/scored_ledger.jsonl --heldout exclude

Features (all are distances: 0 = identical behaviour, larger = more different):
    choice_js    mean Jensen-Shannon divergence between answer distributions on `choice` probes
    choice_mode  fraction of `choice` probes whose most frequent answer differs (robust to decoding strategy)
    bits_l1      mean L1 distance between bit-sequence statistics (P(1), transitions, run lengths, length)
    numbers_l1   mean L1 distance between number histograms (+ adjacent-repeat rate)
    prefix       1 - mean common-prefix ratio of `greedy_long` answers (long shared prefix = same weights)
    mmd          mean unbiased MMD^2 with a character Hamming kernel on the first 64 chars, over all probes
A feature is NaN for a pair when the data needed is missing (e.g. no answers for that probe type yet).
"""
import argparse
import csv
import itertools
import os
import re
from collections import Counter, defaultdict

import numpy as np

import config
from ledger import load_records

FEATURES = ["choice_js", "choice_mode", "bits_l1", "numbers_l1", "prefix", "mmd"]
WORD = re.compile(r"[a-z0-9]+")
MMD_LEN = 64
MAX_RUN = 8


# ---------------------------------------------------------------- loading and parsing
def load_answers(ledger_path, probes: dict, heldout: str = "all") -> dict:
    """{endpoint: {probe_id: [response, ...]}} for answered requests. heldout: all | exclude | only."""
    out = defaultdict(lambda: defaultdict(list))
    for r in load_records(ledger_path):
        if r.get("status") != "ok" or r.get("response") is None:
            continue
        p = probes.get(r["probe_id"])
        if p is None:
            continue
        is_ho = bool(p.get("heldout"))
        if (heldout == "exclude" and is_ho) or (heldout == "only" and not is_ho):
            continue
        out[r["endpoint"]][r["probe_id"]].append(r["response"])
    return out


def parse_choice(text: str, options=None) -> str:
    words = WORD.findall(text.lower())
    if options:
        opts = {o.lower() for o in options}
        for w in words:
            if w in opts:
                return w
        return "<other>"
    return words[0] if words else "<empty>"


def parse_bits(text: str, alphabet: str = "01") -> str:
    zero, one = alphabet[0].upper(), alphabet[1].upper()
    return "".join("0" if c == zero else "1" for c in text.upper() if c in (zero, one))


def parse_numbers(text: str, lo: int, hi: int) -> list:
    return [v for v in (int(x) for x in re.findall(r"-?\d+", text)) if lo <= v <= hi]


# ---------------------------------------------------------------- per-endpoint statistics
def bits_vector(seqs):
    seqs = [s for s in seqs if len(s) >= 10]
    if not seqs:
        return None
    n = n1 = 0
    trans = np.zeros((2, 2))
    runs = np.zeros(MAX_RUN)
    for s in seqs:
        n += len(s)
        n1 += s.count("1")
        for a, b in zip(s, s[1:]):
            trans[int(a), int(b)] += 1
        for m in re.finditer(r"0+|1+", s):
            runs[min(len(m.group()), MAX_RUN) - 1] += 1
    p_1g0 = trans[0, 1] / max(trans[0].sum(), 1)
    p_1g1 = trans[1, 1] / max(trans[1].sum(), 1)
    runs /= max(runs.sum(), 1)
    mean_len = min(np.mean([len(s) for s in seqs]) / 300.0, 1.0)
    return np.concatenate([[n1 / n, p_1g0, p_1g1, mean_len], runs])


def numbers_vector(lists, lo, hi):
    vals = [v for lst in lists for v in lst]
    if len(vals) < 10:
        return None
    hist = np.bincount(np.array(vals) - lo, minlength=hi - lo + 1).astype(float)
    hist /= hist.sum()
    pairs = [(a, b) for lst in lists for a, b in zip(lst, lst[1:])]
    rep = sum(a == b for a, b in pairs) / max(len(pairs), 1)
    return np.concatenate([hist, [rep]])


def profile(answers_ep: dict, probes: dict) -> dict:
    prof = {"choice": {}, "bits": {}, "numbers": {}, "long": {}, "raw": {}, "decode_dup": None, "n": 0}
    dups = []
    for pid, resps in answers_ep.items():
        p = probes[pid]
        prof["raw"][pid] = resps
        prof["n"] += len(resps)
        t = p["type"]
        if t == "decode" and len(resps) >= 2:
            pairs = list(itertools.combinations(resps, 2))
            dups.append(sum(a == b for a, b in pairs) / len(pairs))
        elif t == "choice":
            prof["choice"][pid] = Counter(parse_choice(r, p.get("options")) for r in resps)
        elif t == "bits":
            v = bits_vector([parse_bits(r, p.get("alphabet", "01")) for r in resps])
            if v is not None:
                prof["bits"][pid] = v
        elif t == "numbers":
            v = numbers_vector([parse_numbers(r, p["lo"], p["hi"]) for r in resps], p["lo"], p["hi"])
            if v is not None:
                prof["numbers"][pid] = v
        elif t == "greedy_long":
            prof["long"][pid] = resps
    if dups:
        prof["decode_dup"] = float(np.mean(dups))
    return prof


def decode_mode(prof) -> str:
    d = prof["decode_dup"]
    if d is None:
        return "unknown"
    return "greedy" if d >= 0.999 else "sampling"


# ---------------------------------------------------------------- pair distances
def js(p, q) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    p /= p.sum(); q /= q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def choice_js(pa, pb):
    vals = []
    for pid in pa["choice"].keys() & pb["choice"].keys():
        ca, cb = pa["choice"][pid], pb["choice"][pid]
        keys = sorted(ca.keys() | cb.keys())
        vals.append(js([ca[k] + 0.5 for k in keys], [cb[k] + 0.5 for k in keys]))
    return float(np.mean(vals)) if vals else np.nan


def choice_mode(pa, pb):
    vals = []
    for pid in pa["choice"].keys() & pb["choice"].keys():
        ma = pa["choice"][pid].most_common(1)[0][0]
        mb = pb["choice"][pid].most_common(1)[0][0]
        vals.append(float(ma != mb))
    return float(np.mean(vals)) if vals else np.nan


def vec_l1(da, db):
    vals = [float(np.abs(da[k] - db[k]).sum()) / 2 for k in da.keys() & db.keys()]
    return float(np.mean(vals)) if vals else np.nan


def prefix_dist(pa, pb):
    vals = []
    for pid in pa["long"].keys() & pb["long"].keys():
        ratios = []
        for a in pa["long"][pid]:
            for b in pb["long"][pid]:
                n = min(len(a), len(b))
                ratios.append(len(os.path.commonprefix([a, b])) / n if n else 0.0)
        if ratios:
            vals.append(np.mean(ratios))
    return 1.0 - float(np.mean(vals)) if vals else np.nan


def _encode(resps):
    X = np.zeros((len(resps), MMD_LEN), dtype=np.int64)
    for i, r in enumerate(resps):
        codes = [ord(c) for c in r[:MMD_LEN]]
        X[i, :len(codes)] = codes
    return X


def mmd2(ra, rb) -> float:
    if len(ra) < 2 or len(rb) < 2:
        return np.nan
    X, Y = _encode(ra), _encode(rb)
    kxx = (X[:, None, :] == X[None, :, :]).mean(-1)
    kyy = (Y[:, None, :] == Y[None, :, :]).mean(-1)
    kxy = (X[:, None, :] == Y[None, :, :]).mean(-1)
    n, m = len(X), len(Y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2 * kxy.mean())


def mmd_dist(pa, pb):
    vals = [mmd2(pa["raw"][k], pb["raw"][k]) for k in pa["raw"].keys() & pb["raw"].keys()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


PAIR_FUNCS = {
    "choice_js": choice_js,
    "choice_mode": choice_mode,
    "bits_l1": lambda a, b: vec_l1(a["bits"], b["bits"]),
    "numbers_l1": lambda a, b: vec_l1(a["numbers"], b["numbers"]),
    "prefix": prefix_dist,
    "mmd": mmd_dist,
}


def compute(ledger_path, probes_path=None, heldout="all", endpoints=None):
    """Returns (endpoints, {feature: n x n matrix with NaN diagonal}, {endpoint: profile})."""
    probes = config.load_probes(probes_path)
    answers = load_answers(ledger_path, probes, heldout)
    eps = sorted(endpoints if endpoints is not None else answers.keys())
    profs = {e: profile(answers.get(e, {}), probes) for e in eps}
    n = len(eps)
    feats = {f: np.full((n, n), np.nan) for f in FEATURES}
    for i, j in itertools.combinations(range(n), 2):
        for f in FEATURES:
            v = PAIR_FUNCS[f](profs[eps[i]], profs[eps[j]])
            feats[f][i, j] = feats[f][j, i] = v
    return eps, feats, profs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--heldout", choices=["all", "exclude", "only"], default="all")
    ap.add_argument("--out", default=None, help="write the pair table as CSV")
    args = ap.parse_args(argv)

    eps, feats, profs = compute(args.ledger, args.probes, args.heldout)
    if not eps:
        raise SystemExit(f"No answered requests in {args.ledger}.")
    print(f"{'endpoint':>9} {'answers':>7} {'decode':>9} {'dup':>5}  top choice answers")
    for e in eps:
        p = profs[e]
        tops = " ".join(f"{k}={c.most_common(1)[0][0]}" for k, c in list(p["choice"].items())[:4])
        dup = "" if p["decode_dup"] is None else f"{p['decode_dup']:.2f}"
        print(f"{e:>9} {p['n']:>7} {decode_mode(p):>9} {dup:>5}  {tops}")

    rows = [(eps[i], eps[j], *[feats[f][i, j] for f in FEATURES]) for i, j in itertools.combinations(range(len(eps)), 2)]
    print("\n" + f"{'a':>7} {'b':>7} " + " ".join(f"{f:>11}" for f in FEATURES))
    for r in rows[:60]:
        print(f"{r[0]:>7} {r[1]:>7} " + " ".join(f"{v:11.4f}" for v in r[2:]))
    if len(rows) > 60:
        print(f"... {len(rows) - 60} more pairs (use --out)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", *FEATURES])
            w.writerows(rows)
        print(f"Pair table written to {args.out}")


if __name__ == "__main__":
    main()
