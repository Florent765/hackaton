"""Shared settings for the model-equality code (TASK_PLAN_v2). Every path can be overridden by an environment variable.

Data paths are relative to the directory you run the scripts from. On JURECA, run from the task root
($TEAM_DIR/model_equality) so that the ledgers in data/ are shared by the whole team.
"""
import json
import os
import sys
from pathlib import Path

SERVER_URL = os.environ.get("ME_SERVER_URL", "http://34.147.229.55").rstrip("/")
TASK_ID = "33-model-equality"                      # fixed by the organizers
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
SUBMIT_URL = f"{SERVER_URL}/submit/{TASK_ID}"

HERE = Path(__file__).resolve().parent             # folder holding the code, probes.json and configs/
DATA_DIR = Path(os.environ.get("ME_DATA_DIR", "data"))
REF_LEDGER = DATA_DIR / "ref_ledger.jsonl"         # reference endpoints ref-01..ref-09 (no query limit)
SCORED_LEDGER = DATA_DIR / "scored_ledger.jsonl"   # scored endpoints ep-01..ep-33 (100 answers each, ever)
SUBMISSIONS_DIR = Path(os.environ.get("ME_SUBMISSIONS_DIR", "submissions"))
PROBES_FILE = HERE / "probes.json"
PRACTICE_ANSWERS = HERE / "practice_answers.csv"

MAX_TOKENS = 256          # server cap per answer
PARALLEL = 4              # server cap on concurrent requests per team (all teammates together)
COOLDOWN_S = 30 * 60      # between submissions, failed ones included
N_SCORED = 33


def api_key() -> str:
    key = os.environ.get("HACKATHON_API_KEY", "").strip()
    if not key:
        sys.exit("Set the team API key first:  export HACKATHON_API_KEY=...   (never write it in a file)")
    return key


def load_probes(path=None) -> dict:
    """{probe_id: probe dict} from probes.json, in file order."""
    data = json.loads(Path(path or PROBES_FILE).read_text(encoding="utf-8"))
    probes = {}
    for p in data["probes"]:
        if p["id"] in probes:
            sys.exit(f"duplicate probe id {p['id']} in probes file")
        probes[p["id"]] = p
    return probes


def scored_endpoints() -> list:
    return [f"ep-{i:02d}" for i in range(1, N_SCORED + 1)]
