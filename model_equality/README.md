# Model equality (task 33)

Code for the model-equality task, following [TASK_PLAN_v2.md](TASK_PLAN_v2.md). Group the 33 scored endpoints `ep-01` … `ep-33` by base model (family) and by exact model (checkpoint + quantization + LoRA), using at most 100 answers per endpoint.

No GPU is needed. Everything runs on a laptop or on the JURECA login node.

## Facts from the starter kit (verified 2026-11-21)

From <https://huggingface.co/datasets/SprintML/model_equality>:

- **Reference endpoints `ref-01` … `ref-09` have no query limit and are never scored.** Use them freely to tune.
- All 9 reference endpoints are **one family** (`P`) served by a separate small model. Their identities:

  | Model | Endpoints (disguise) |
  |---|---|
  | S | ref-01 plain, ref-03 system prompt, ref-05 persona, ref-06 watermark, ref-08 sampling at T=1.0 |
  | T1 | ref-02 fp8 quantization |
  | T2 / T3 / T4 | ref-09 / ref-04 / ref-07 LoRA at strength 0.3 / 0.7 / 1.5 |

  The references can therefore tune the **model** split (90 % of the score) but not the family split.
- API: `GET {server}/33-model-equality/v1/models` returns the budget, and `POST …/v1/chat/completions` takes `{"model": ep, "max_tokens": ≤256, "messages": [{"role": "user", ...}]}`. Answers are sampled by default.
- HTTP 429 with a `Retry-After` header means the server is busy and the request was not charged. A 429 without that header means the endpoint's budget is used up.
- Submission: `POST {server}/submit/33-model-equality` with a CSV file, then a 30-minute cooldown. Failed submissions also start the cooldown.

## Files

| File | Role |
|---|---|
| `config.py` | Server URL, paths, limits. API key read from `HACKATHON_API_KEY`. |
| `ledger.py` | Append-only JSONL ledger: one `pending` line before each request, one outcome line after. |
| `query.py` | `budget`, `run`, `reconcile`. Ledger-backed client, 4 requests at a time, never retries a scored request blindly. |
| `probes.json` | Probe bank: `decode`, `choice`, `bits`, `numbers`, `greedy_long`; `heldout` variants are kept out of tuning. |
| `features.py` | Parses answers and computes 6 pairwise distances (`choice_js`, `choice_mode`, `bits_l1`, `numbers_l1`, `prefix`, `mmd`). |
| `cluster.py` | `tune` on reference endpoints, `solve` to write a submission (two passes: family, then model). |
| `validate_submission.py` | Checks header, the 33 endpoints, labels, and model-to-family nesting. |
| `submit.py` | Validates, checks the cooldown, submits, and logs to `submissions/submit_log.csv`. |
| `configs/default.json` | Default weights and thresholds. `tune` writes `configs/tuned.json`. |
| `score_practice.py`, `practice_answers.csv` | Organizer files, unchanged. |

Generated, and kept out of git: `data/` (ledgers) and `submissions/` (CSVs, sidecars, log).

## Setup

**JURECA** (from your laptop, at the repo root, then on JURECA):

```bash
# laptop: upload the code
scp model_equality/*.py model_equality/*.json model_equality/*.csv \
    jureca:/p/scratch/training2666/delorme2/team51/model_equality/work/$JSC_USER/
scp -r model_equality/configs jureca:/p/scratch/training2666/delorme2/team51/model_equality/work/$JSC_USER/

# JURECA: run from the task root so data/ and submissions/ are shared by the team
jutil env activate -p training2666
cd /p/scratch/training2666/delorme2/team51/model_equality
source .venv/bin/activate          # needs requests, numpy, scipy (pip install -r work/$USER/requirements.txt if missing)
export HACKATHON_API_KEY=...       # in the shell only, never in a file
```

On JURECA, prefix the commands below with `work/$USER/`, for example `python work/$USER/query.py budget`.

**Laptop:** `pip install -r requirements.txt`, set `HACKATHON_API_KEY`, and run the commands from inside `model_equality/`. Data then stays on your laptop, so agree with the team where the shared ledger lives.

## Workflow

### 1. Check the budget (free)
```bash
python query.py budget
```
This shows used and left answers per endpoint, and flags endpoints where the server's counter and our ledger differ. A difference means someone queried outside the ledger.

### 2. Collect reference answers (free, unlimited)
```bash
python query.py run --target ref --repeats 10
```
This sends all probes to `ref-01` … `ref-09`, 10 times each (about 1,700 answers). Running it again fills only what is missing. Raise `--repeats` to add more.

### 3. Look at the features and tune
```bash
python features.py data/ref_ledger.jsonl                 # decode mode per endpoint + pair distances
python cluster.py tune                                   # weights x thresholds, practice MCC, writes configs/tuned.json
python cluster.py solve --target ref --config configs/tuned.json    # same pipeline as for scored, on ref-*
python score_practice.py submissions/practice_*.csv      # organizer scorer, cross-check
```
`tune` prints the MCC on the tuning probes and on the held-out probes. With only 36 pairs, pick a configuration that is good on **both**.

Worth checking by hand: does `ref-02` (fp8) or `ref-09` (LoRA 0.3) separate from the S group, and do `ref-06` (watermark) and `ref-08` (T=1.0) stay inside it?

If they don't, improve the probes (add new ids to `probes.json`) and repeat steps 2–3. That costs nothing.

### 4. Spend scored budget, in stages
Scored runs only print a plan unless you add `--yes`. `--max-total N` caps the answers in the ledger at N per endpoint, across all probes, so running the same command twice spends nothing more.

```bash
# first batch: about 40 answers per endpoint (TASK_PLAN_v2 step 7)
python query.py run --target scored --repeats 2 --max-total 40          # read the plan
python query.py run --target scored --repeats 2 --max-total 40 --yes    # spend
python query.py reconcile                                               # if any request was marked ambiguous
```
Rules:
- **One person** runs scored queries at a time. The ledger has a lock file, but the server's limit of 4 concurrent requests applies to the whole team.
- A scored request that hits a timeout or a 5xx is marked `ambiguous` and **not retried**. `reconcile` compares the server's counters with the ledger and marks it `void` (not charged, can be asked again) or `lost` (charged).
- Never edit the prompt of a probe id that has already been sent to scored endpoints. Add a new id instead.

### 5. Cluster and submit
```bash
python cluster.py solve --config configs/tuned.json          # writes submissions/sub_<time>.csv + .json sidecar
python submit.py submissions/sub_<time>.csv --note "tuned all_equal, 40/endpoint"          # dry run
python submit.py submissions/sub_<time>.csv --note "tuned all_equal, 40/endpoint" --yes    # sends
```
`solve` prints the clusters. Read them before submitting. The `.json` sidecar keeps the config, its hash, the ledger hash and the decode mode of each endpoint.

### 6. Spend the reserve (60 answers per endpoint)
Look in the sidecar and the pair table (`python features.py data/scored_ledger.jsonl --out data/scored_pairs.csv`) for pairs close to the threshold. Then add repeats on the probes that separate them best, on those endpoints only:
```bash
python query.py run --target scored --endpoints ep-04,ep-16 --types choice,bits --repeats 4 --max-total 60 --yes
```

## Testing without the real server

The code was tested end to end against a local mock of the API: querying, the "ambiguous → lost" path, idempotent re-runs, tuning, solving, scoring and a submit dry run. `ME_SERVER_URL` points the code at another server. The mock is easy to separate, so a perfect MCC on it says nothing about the real task.

## Known limits

- Distances are z-scored over the endpoints being clustered, so a threshold tuned on the 9 references may not carry over exactly to the 33 scored endpoints. Check the cluster sizes before submitting.
- `"auto"` thresholds cut at the largest gap and therefore always produce at least two clusters.
- The family pass cannot be tuned on references (one family). It uses `"auto"` by default.
- The `prefix` feature only means something for endpoints that decode greedily or at low temperature. `features.py` prints the decode mode of each endpoint.
- The probes are a first draft. Expect to replace some after looking at real reference answers, for example if a persona endpoint refuses the strict formats.
