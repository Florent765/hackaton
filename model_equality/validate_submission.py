"""Check a submission CSV before sending it (a failed submission also costs the 30-minute cooldown).

    python validate_submission.py submissions/sub_XXXX.csv
    python validate_submission.py practice.csv --practice        # ref-01..ref-09 instead of ep-01..ep-33

Checks: header endpoint,family,model (optional), exactly the expected endpoints once each, 3 non-empty columns,
no commas inside labels, and every model label belongs to a single family label.
"""
import argparse
import csv
import sys
from pathlib import Path

import config


def expected(practice: bool) -> list:
    return [f"ref-{i:02d}" for i in range(1, 10)] if practice else config.scored_endpoints()


def validate(path, practice: bool = False) -> list:
    errors = []
    path = Path(path)
    if not path.is_file():
        return [f"file not found: {path}"]
    if path.suffix.lower() != ".csv":
        errors.append("file must end with .csv")
    raw = path.read_text(encoding="utf-8-sig")
    rows = [r for r in csv.reader(raw.splitlines()) if any(c.strip() for c in r)]
    if rows and [c.strip() for c in rows[0]] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    seen, model_family = {}, {}
    for k, r in enumerate(rows, 1):
        if len(r) != 3:
            errors.append(f"line {k}: expected 3 columns, got {len(r)}: {r}")
            continue
        ep, fam, mod = (c.strip() for c in r)
        if not fam or not mod:
            errors.append(f"line {k}: empty label for {ep}")
        if ep in seen:
            errors.append(f"{ep} appears twice")
        seen[ep] = (fam, mod)
        if mod in model_family and model_family[mod] != fam:
            errors.append(f"model label {mod} used in families {model_family[mod]} and {fam}")
        model_family.setdefault(mod, fam)
    want = expected(practice)
    missing = [e for e in want if e not in seen]
    extra = [e for e in seen if e not in want]
    if missing:
        errors.append(f"missing endpoints: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected endpoints: {', '.join(extra)}")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--practice", action="store_true")
    args = ap.parse_args(argv)
    errors = validate(args.file, args.practice)
    if errors:
        print("INVALID:\n  " + "\n  ".join(errors))
        return 1
    print(f"OK: {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
