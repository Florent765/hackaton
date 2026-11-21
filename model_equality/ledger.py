"""Append-only query ledger (TASK_PLAN_v2 step 1).

Every request gets a random id (rid). A "pending" line is written BEFORE the request is sent, and a second line with the
outcome after. The state of a request is the merge of all its lines, last one wins. Statuses:

    pending    sent (or about to be), no outcome recorded: the process died. Counts as spent until reconciled.
    ok         answer received and stored.
    error      the server refused the request (not charged), or the endpoint budget was exhausted.
    ambiguous  network error / timeout / 5xx after sending: unknown whether it was charged. Counts as spent.
    void       reconciled: was not charged. Can be queried again.
    lost       reconciled: was charged but no answer was stored.
"""
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

SPENT = {"ok", "pending", "ambiguous", "lost"}     # statuses that use (or may use) budget
UNRESOLVED = {"pending", "ambiguous"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_records(path) -> list:
    """Current state of every request in the ledger (one dict per rid)."""
    path = Path(path)
    if not path.exists():
        return []
    state = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:      # torn last line after a crash
                continue
            state.setdefault(rec["rid"], {}).update(rec)
    return list(state.values())


class Ledger:
    """Thread-safe appender with an exclusive lock file, so two processes never write the same ledger."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile = self.path.with_name(self.path.name + ".lock")
        self._lock = threading.Lock()

    def __enter__(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            sys.exit(f"{self.lockfile} exists: someone is already using this ledger. "
                     f"If you are sure nobody is, delete the lock file and retry.")
        os.write(fd, f"pid={os.getpid()} user={os.environ.get('USER', os.environ.get('USERNAME', '?'))} {now()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            self.lockfile.unlink()
        except FileNotFoundError:
            pass

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
