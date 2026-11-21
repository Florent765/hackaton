"""Validate, submit and log one model-equality CSV (TASK_PLAN_v2 step 10). Same server call as the organizers'
submission.py, with the key read from the environment.

    export HACKATHON_API_KEY=...
    python submit.py submissions/sub_XXXX.csv --note "tuned all_but_mmd, 40 answers/endpoint"          # dry run
    python submit.py submissions/sub_XXXX.csv --note "tuned all_but_mmd, 40 answers/endpoint" --yes    # sends

The server enforces a 30-minute cooldown between submissions, failed ones included. This script also refuses to send
within 30 minutes of the last submission in its own log; teammates submitting from elsewhere are not visible to it.
Every attempt is appended to submissions/submit_log.csv.
"""
import argparse
import csv
import getpass
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
from validate_submission import validate

FIELDS = ["time", "user", "file", "sha256", "note", "status", "score", "previous_best", "improved", "message"]


def log_path() -> Path:
    return config.SUBMISSIONS_DIR / "submit_log.csv"


def last_submission_time():
    p = log_path()
    if not p.exists():
        return None
    times = [r["time"] for r in csv.DictReader(p.open(newline="")) if r.get("status") not in ("invalid", "unreachable")]
    return datetime.fromisoformat(times[-1]) if times else None


def log(row: dict) -> None:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--note", required=True, help="what this partition is (method, config, evidence)")
    ap.add_argument("--yes", action="store_true", help="really submit")
    args = ap.parse_args(argv)

    base = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "user": getpass.getuser(),
            "file": str(args.file), "note": args.note}
    errors = validate(args.file)
    if errors:
        log({**base, "status": "invalid", "message": "; ".join(errors)})
        print("Not submitted, invalid file:\n  " + "\n  ".join(errors))
        return 1
    sha = hashlib.sha256(args.file.read_bytes()).hexdigest()
    base["sha256"] = sha

    last = last_submission_time()
    if last is not None:
        wait = config.COOLDOWN_S - (datetime.now(timezone.utc) - last).total_seconds()
        if wait > 0:
            print(f"Cooldown: last submission at {last.isoformat()}, wait {wait / 60:.1f} more minutes.")
            return 1
    print(f"File {args.file} valid, sha256 {sha[:12]}.")
    if not args.yes:
        print("Dry run. Add --yes to submit (starts the 30-minute cooldown for the whole team).")
        return 0

    print("Submitting. Scoring can take a few minutes...")
    try:
        with args.file.open("rb") as f:
            reply = requests.post(config.SUBMIT_URL, headers={"X-API-Key": config.api_key()},
                                  files={"file": (args.file.name, f)}, timeout=(600, 600))
    except requests.RequestException as exc:
        log({**base, "status": "unreachable", "message": str(exc)[:300]})
        print(f"Could not reach the server: {exc}", file=sys.stderr)
        return 1
    try:
        body = reply.json()
    except ValueError:
        body = {"status": f"http_{reply.status_code}", "message": reply.text[:300]}
    if reply.status_code != 200:
        body = {"status": f"rejected_{reply.status_code}", "message": str(body.get("detail", body))}
    log({**base, **{k: body.get(k, "") for k in ("status", "score", "previous_best", "improved", "message")}})
    print(f"status={body.get('status')} score={body.get('score')} previous_best={body.get('previous_best')} "
          f"improved={body.get('improved')}")
    if body.get("message"):
        print(body["message"])
    return 0 if body.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
