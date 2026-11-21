"""Query the endpoints through the ledger (TASK_PLAN_v2 steps 1, 4, 7, 9).

    python query.py budget                                   # answers used / left per endpoint, ledger state
    python query.py run --target ref --repeats 10            # reference endpoints: no limit, safe to repeat
    python query.py run --target scored --types decode --repeats 3 --max-total 3          # dry run: prints plan
    python query.py run --target scored --types decode --repeats 3 --max-total 3 --yes    # spends budget
    python query.py reconcile                                # resolve pending/ambiguous scored requests

Rules built in:
  * one (endpoint, probe, repeat) is asked at most once; re-running only fills what is missing;
  * --repeats N means repeats 0..N-1 (raising N later adds new repeats only);
  * scored runs need --max-total N: the ledger may hold at most N spent answers per endpoint after the run (all probes
    together), so running the same command twice spends nothing more; and --yes, otherwise only the plan prints;
  * scored requests are NEVER retried after a network error / timeout / 5xx: they become "ambiguous" and count as
    spent until `reconcile` compares the server's counters with the ledger;
  * at most 4 requests at once (team-wide cap: make sure nobody else queries at the same time).
"""
import argparse
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

import config
from ledger import SPENT, UNRESOLVED, Ledger, load_records, now


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["X-API-Key"] = config.api_key()
    return s


def get_budget(session) -> dict:
    """{endpoint: (used, limit)} as the server reports it."""
    r = session.get(f"{config.BASE_URL}/models", timeout=60)
    if r.status_code != 200:
        sys.exit(f"Budget query failed: HTTP {r.status_code}: {r.text[:300]}")
    return {m["id"]: (m["budget"]["used"], m["budget"]["limit"]) for m in r.json()["data"]}


def detail_of(r) -> str:
    if r.headers.get("content-type", "").startswith("application/json"):
        try:
            return str(r.json().get("detail", r.text[:300]))
        except ValueError:
            pass
    return r.text[:300]


# ---------------------------------------------------------------- budget
def cmd_budget(args) -> None:
    budget = get_budget(make_session())
    recs = load_records(args.ledger or config.SCORED_LEDGER)
    ok = Counter(r["endpoint"] for r in recs if r.get("status") == "ok")
    unres = Counter(r["endpoint"] for r in recs if r.get("status") in UNRESOLVED)
    print(f"{'endpoint':>9} {'used':>5} {'limit':>6} {'left':>5} {'ledger_ok':>9} {'unresolved':>10}")
    for ep, (used, limit) in budget.items():
        flag = "" if not ep.startswith("ep-") or used == ok[ep] + unres[ep] else "   <- server and ledger differ"
        print(f"{ep:>9} {used:>5} {limit:>6} {limit - used:>5} {ok[ep]:>9} {unres[ep]:>10}{flag}")


# ---------------------------------------------------------------- run
def plan_jobs(args, budget):
    probes = config.load_probes(args.probes)
    types = set(args.types.split(",")) if args.types else None
    ids = set(args.ids.split(",")) if args.ids else None
    sel = [p for p in probes.values()
           if (types is None or p["type"] in types) and (ids is None or p["id"] in ids)
           and not (args.skip_heldout and p.get("heldout"))]
    if not sel:
        sys.exit("No probe matches --types/--ids.")

    scored = args.target == "scored"
    prefix = "ep-" if scored else "ref-"
    known = [e for e in budget if e.startswith(prefix)]
    eps = args.endpoints.split(",") if args.endpoints else known
    bad = [e for e in eps if e not in known]
    if bad:
        sys.exit(f"Unknown endpoints {bad}; valid: {', '.join(known)}")

    recs = load_records(args.ledger_path)
    done_status = SPENT if scored else {"ok"}         # reference endpoints: re-ask anything not answered
    done = {(r["endpoint"], r["probe_id"], r["repeat"]) for r in recs if r.get("status") in done_status}
    unres = Counter(r["endpoint"] for r in recs if r.get("status") in UNRESOLVED)
    spent = Counter(r["endpoint"] for r in recs if r.get("status") in SPENT)

    cap = {}
    for e in eps:
        used, limit = budget[e]
        left = limit - used - (unres[e] if scored else 0)   # conservative: unresolved may still be charged
        room = args.max_total - spent[e] if args.max_total is not None else left
        cap[e] = max(0, min(left, room))

    jobs, planned, over = [], Counter(), 0
    for rep in range(args.repeats):                    # round-robin: an early stop leaves endpoints balanced
        for p in sel:
            for e in eps:
                if (e, p["id"], rep) in done:
                    continue
                if planned[e] < cap[e]:
                    planned[e] += 1
                    jobs.append((e, p, rep))
                else:
                    over += 1
    return jobs, planned, over, sel, eps


def cmd_run(args) -> None:
    scored = args.target == "scored"
    if scored and args.max_total is None:
        sys.exit("--max-total is required for scored endpoints (each answer is gone for good).")
    args.ledger_path = args.ledger or (config.SCORED_LEDGER if scored else config.REF_LEDGER)
    session = make_session()
    budget = get_budget(session)
    jobs, planned, over, sel, eps = plan_jobs(args, budget)

    by_type = Counter(p["type"] for _, p, _ in jobs)
    print(f"Target {args.target}: {len(sel)} probes x {len(eps)} endpoints x {args.repeats} repeats")
    print(f"  to query now: {len(jobs)}  by type: {dict(by_type)}")
    if planned:
        print(f"  per endpoint: min {min(planned[e] for e in eps)}, max {max(planned[e] for e in eps)}")
    if over:
        print(f"  left out (budget or --max-total cap): {over}")
    print(f"  ledger: {args.ledger_path}")
    if not jobs:
        print("Nothing to do.")
        return
    if scored and not args.yes:
        print("\nDry run. Re-run with --yes to spend these answers on the scored endpoints.")
        return

    exhausted, counter, t0 = set(), [0], time.time()

    with Ledger(args.ledger_path) as ledger:
        def one(job):
            ep, p, rep = job
            if ep in exhausted:
                return
            rid = uuid.uuid4().hex
            ledger.append({"rid": rid, "status": "pending", "endpoint": ep, "probe_id": p["id"], "type": p["type"],
                           "variant": p.get("variant", ""), "repeat": rep, "prompt": p["prompt"],
                           "max_tokens": config.MAX_TOKENS, "t_sent": now()})
            tries = 0
            while True:
                try:
                    r = session.post(f"{config.BASE_URL}/chat/completions", timeout=180,
                                     json={"model": ep, "max_tokens": config.MAX_TOKENS,
                                           "messages": [{"role": "user", "content": p["prompt"]}]})
                except requests.RequestException as exc:
                    if scored:
                        ledger.append({"rid": rid, "status": "ambiguous", "error": type(exc).__name__, "t_done": now()})
                        print(f"{ep} {p['id']} r{rep}: {type(exc).__name__}, marked ambiguous (not retried)")
                        return
                    tries += 1
                    if tries > 5:
                        ledger.append({"rid": rid, "status": "error", "error": type(exc).__name__, "t_done": now()})
                        return
                    time.sleep(15)
                    continue
                if r.status_code == 200:
                    body = r.json()
                    ledger.append({"rid": rid, "status": "ok", "response": body["choices"][0]["message"]["content"],
                                   "budget": body.get("budget"), "t_done": now()})
                    counter[0] += 1
                    if counter[0] % 20 == 0 or counter[0] == len(jobs):
                        print(f"{counter[0]}/{len(jobs)} answers ({time.time() - t0:.0f}s)")
                    return
                if r.status_code == 429 and "Retry-After" in r.headers:      # busy, not charged: wait for a slot
                    time.sleep(int(r.headers["Retry-After"]))
                    continue
                if r.status_code == 429:                                      # endpoint budget used up
                    exhausted.add(ep)
                    ledger.append({"rid": rid, "status": "error", "http": 429, "error": detail_of(r), "t_done": now()})
                    print(f"{ep}: budget exhausted ({detail_of(r)})")
                    return
                if r.status_code in (502, 503, 504):
                    if scored:
                        ledger.append({"rid": rid, "status": "ambiguous", "http": r.status_code, "t_done": now()})
                        print(f"{ep} {p['id']} r{rep}: HTTP {r.status_code}, marked ambiguous (not retried)")
                        return
                    time.sleep(15)
                    continue
                ledger.append({"rid": rid, "status": "error", "http": r.status_code, "error": detail_of(r),
                               "t_done": now()})
                print(f"{ep} {p['id']} r{rep}: HTTP {r.status_code}: {detail_of(r)}", file=sys.stderr)
                return

        with ThreadPoolExecutor(max_workers=config.PARALLEL) as pool:
            list(pool.map(one, jobs))
    print(f"Finished: {counter[0]} answers stored in {args.ledger_path}")


# ---------------------------------------------------------------- reconcile
def cmd_reconcile(args) -> None:
    path = args.ledger or config.SCORED_LEDGER
    budget = get_budget(make_session())
    recs = load_records(path)
    with Ledger(path) as ledger:
        for ep in sorted({r["endpoint"] for r in recs}):
            ok = sum(1 for r in recs if r["endpoint"] == ep and r.get("status") == "ok")
            lost = sum(1 for r in recs if r["endpoint"] == ep and r.get("status") == "lost")
            unres = [r for r in recs if r["endpoint"] == ep and r.get("status") in UNRESOLVED]
            if not unres:
                continue
            used = budget.get(ep, (None,))[0]
            if used == ok + lost:
                new = "void"        # none of them was charged
            elif used is not None and used >= ok + lost + len(unres):
                new = "lost"        # all charged (or someone else also queried: check with the team)
            else:
                print(f"{ep}: server used={used}, ledger ok={ok} lost={lost} unresolved={len(unres)}: "
                      f"partial match, left unresolved")
                continue
            for r in unres:
                ledger.append({"rid": r["rid"], "status": new, "t_done": now(), "reconciled_used": used})
            print(f"{ep}: {len(unres)} unresolved -> {new}")
    print("Reconcile done. (Assumes every scored query of the team goes through this ledger.)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="server counters vs ledger")
    b.add_argument("--ledger", default=None)
    b.set_defaults(func=cmd_budget)

    r = sub.add_parser("run", help="query probes")
    r.add_argument("--target", choices=["ref", "scored"], required=True)
    r.add_argument("--types", default=None, help="comma list: decode,choice,bits,numbers,greedy_long")
    r.add_argument("--ids", default=None, help="comma list of probe ids")
    r.add_argument("--endpoints", default=None, help="comma list (default: all of the target)")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--max-total", type=int, default=None,
                   help="scored: max spent answers per endpoint in the ledger after this run (all probes together)")
    r.add_argument("--skip-heldout", action="store_true", help="do not query held-out probe variants")
    r.add_argument("--probes", default=None, help="probe file (default: probes.json next to this script)")
    r.add_argument("--ledger", default=None)
    r.add_argument("--yes", action="store_true", help="really spend scored budget")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("reconcile", help="resolve pending/ambiguous scored requests")
    c.add_argument("--ledger", default=None)
    c.set_defaults(func=cmd_reconcile)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
