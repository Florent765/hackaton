# Model equality (task 33)

Code for the model-equality task. Group the 33 scored endpoints `ep-01` … `ep-33` by base model (family) and by exact model (checkpoint + quantization + LoRA), using at most 100 answers per endpoint. The background is in [TASK_PLAN_v2.md](TASK_PLAN_v2.md); this README is the reference for running the code.

No GPU is needed. Everything runs on a laptop or on the JURECA login node.

## Facts from the starter kit (verified 2026-11-21)

From <https://huggingface.co/datasets/SprintML/model_equality>:

- **Reference endpoints `ref-01` … `ref-09` have no query limit and are never scored.** All tuning happens on them.
- All 9 references are **one family** (`P`) served by a separate small model:

  | Model | Endpoints (disguise) |
  |---|---|
  | S | ref-01 plain, ref-03 system prompt, ref-05 persona, ref-06 watermark, ref-08 sampling at T=1.0 |
  | T1 | ref-02 fp8 quantization |
  | T2 / T3 / T4 | ref-09 / ref-04 / ref-07 LoRA at strength 0.3 / 0.7 / 1.5 |

  They can therefore tune the **model** split (90 % of the score) but not the family split.
- API: `GET {server}/33-model-equality/v1/models` returns the budget, and `POST …/v1/chat/completions` takes `{"model": ep, "max_tokens": ≤256, "messages": [{"role": "user", ...}]}`. Answers are sampled.
- HTTP 429 **with** `Retry-After` means the server is busy and the request was not charged. A 429 **without** it means the endpoint's budget is used up.
- Submission: `POST {server}/submit/33-model-equality` with a CSV file, then a 30-minute cooldown. Failed submissions also start the cooldown.

## How it works

1. **Calibrated evidence for each pair.** For every pair of endpoints and every feature, the answers of both endpoints to the same probe are pooled and randomly re-split (permutation test). The feature value is a z-score: about 0 when the two endpoints are indistinguishable, well above 0 when they clearly differ. The scale does not depend on which other endpoints are present, so a threshold tuned on the references carries over to the scored endpoints, **as long as each probe gets the same number of answers**.
2. **The budget goes to the probes that work.** `cluster.py tune` ranks every probe by how well it separates different models from disguised copies of one model. It then tests "K best probes × (90 ÷ K) answers each" with leave-one-block-out cross-validation, on blocks of reference answers of exactly that size. The winning plan goes to `configs/tuned.json`.
3. **Two-pass clustering.** Families are cut first, on the same answers, at the largest gap between merges. Models are then cut inside each family at the tuned threshold.

### Results on a local mock (not the real server)

The mock has subtle fp8 and LoRA 0.3 differences, a persona that chats and sometimes ignores the format, a watermark that flips symbols, and a higher temperature.

| Setup | Result |
|---|---|
| 19 probes × 3 answers each | MCC 0.55 on the references; the pipeline refuses to spend scored budget |
| 3 best probes × 30 answers (this workflow) | MCC 1.0 on the references (cross-validated); **task score 0.725 and 0.738** on two independent simulations of the 33 scored endpoints (family MCC 0.40, model MCC 0.76–0.78) |

These numbers show the method works on the mock, not what the real score will be.

## Files

| File | Role |
|---|---|
| `config.py` | Server URL, paths, limits. API key read from `HACKATHON_API_KEY`. |
| `ledger.py` | Append-only JSONL ledger: one `pending` line before each request, one outcome line after. |
| `query.py` | `budget`, `run`, `reconcile`. Ledger-backed client, 4 requests at a time, never retries a scored request blindly. `--recommended` queries only the probes chosen by `tune`. |
| `probes.json` | Probe bank: `decode`, `choice`, `bits`, `numbers`, `greedy_long`. |
| `features.py` | Parsing, parse-failure diagnostics, permutation z-scores for 5 features (`choice`, `bits`, `numbers`, `prefix`, `mmd`). |
| `cluster.py` | `tune` (probe ranking, allocation search, threshold) and `solve` (writes a submission). |
| `validate_submission.py` | Checks header, the 33 endpoints, labels, and model-to-family nesting. |
| `submit.py` | Validates, checks the cooldown, submits, and logs to `submissions/submit_log.csv`. |
| `configs/default.json` | Starting config. `tune` writes `configs/tuned.json`. |
| `score_practice.py`, `practice_answers.csv` | Organizer files, unchanged. |

Generated, and kept out of git: `data/` (ledgers) and `submissions/` (CSVs, sidecars, log).

## Setup

**JURECA** (from your laptop at the repo root, then on JURECA):

```bash
# laptop: upload the code
scp model_equality/*.py model_equality/*.json model_equality/*.csv \
    jureca:/p/scratch/training2666/delorme2/team51/model_equality/work/$JSC_USER/
scp -r model_equality/configs jureca:/p/scratch/training2666/delorme2/team51/model_equality/work/$JSC_USER/

# JURECA: run from the task root so data/ and submissions/ are shared by the team
jutil env activate -p training2666
cd /p/scratch/training2666/delorme2/team51/model_equality
source .venv/bin/activate          # has requests, numpy, scipy (from the organizers' requirements.txt)
export HACKATHON_API_KEY=...       # in the shell only, never in a file
```

On JURECA, prefix the commands below with `work/$USER/`, for example `python work/$USER/query.py budget`, and `configs/` with `work/$USER/configs/`.

**Laptop:** `pip install -r requirements.txt`, set `HACKATHON_API_KEY`, and run the commands from inside `model_equality/`. Agree with the team on where the shared ledger lives.

## Workflow

Steps 1 to 4 only use the free reference endpoints. **Do not spend scored budget before step 4 gives a cross-validated MCC ≥ 0.8.**

### 0. Check the budget (free)
```bash
python query.py budget
```
Shows used and remaining answers per endpoint. It flags endpoints where the server's counter and the ledger differ, which means someone queried outside the ledger.

### 1. Quick reference collection (free)
```bash
python query.py run --target ref --repeats 6
```
Every probe, 6 times, on the 9 references: about 1,000 requests. The command resumes where it stopped if interrupted.

### 2. Diagnostics and first tune
```bash
python features.py data/ref_ledger.jsonl --repeats 0:3     # parse failures, decode mode, pair z-scores
python cluster.py tune
```
- **Parse failures above 30 %** mean an endpoint ignores a probe's format (often the persona). Reword the probe under a **new id** in `probes.json` and collect it again.
- **Probe ranking:** an AUC near 1 means the probe separates different models from disguised copies of one model; 0.5 means it is useless.
- **Allocation search:** at this stage most plans cannot be tested yet. For each one, `tune` prints the reference collection it needs.

### 3. Deep reference collection (free, but long)
Run the command `tune` printed for the plans you want to test, for example:
```bash
python query.py run --target ref --ids g_lighthouse,g_station,g_fridge --repeats 60
```
This can mean several thousand requests at 4 at a time: **start it early**. It can be stopped and resumed.

### 4. Tune again and check
```bash
python cluster.py tune
python cluster.py solve --target ref --config configs/tuned.json --repeats 0:R    # R = repeats_per_probe in tuned.json
python score_practice.py submissions/practice_<time>.csv                       # organizer scorer, cross-check
```
`tune` writes `configs/tuned.json` (probes, answers per probe R, weights, threshold) and prints the exact scored collection command. Go on only if the cross-validated MCC is at least 0.8. Otherwise `tune` prints a warning: add or reword probes, collect more reference repeats, and tune again.

### 5. Scored collection with the recommended probes
Without `--yes`, the command only prints the plan. `--max-total` caps the answers per endpoint in the ledger, so running the same command twice spends nothing more.
```bash
python query.py run --target scored --recommended configs/tuned.json --repeats R --max-total K*R          # read the plan
python query.py run --target scored --recommended configs/tuned.json --repeats R --max-total K*R --yes    # spend
python query.py reconcile        # only if some request was marked ambiguous
```
Replace R and K×R with the values `tune` printed (90 answers or slightly fewer; 10 stay in reserve).

Rules:
- **One person** runs scored queries at a time. The ledger has a lock file, but the server's limit of 4 concurrent requests applies to the whole team.
- A scored request that times out or gets a 5xx is marked `ambiguous` and is **not retried**. `reconcile` compares the server's counters with the ledger and marks it `void` (not charged, can be asked again) or `lost` (charged).
- Never edit the prompt of a probe id that has already been sent to scored endpoints. Add a new id instead.
- Keep R the same as in `tuned.json`. The threshold is only valid for that number of answers per probe.

### 6. Cluster and submit
```bash
python cluster.py solve --config configs/tuned.json      # writes submissions/sub_<time>.csv + .json sidecar
python submit.py submissions/sub_<time>.csv --note "tuned 3x30 only_prefix t=3"           # dry run
python submit.py submissions/sub_<time>.csv --note "tuned 3x30 only_prefix t=3" --yes     # sends
```
`solve` prints the clusters, their sizes, and the pairs closest to the threshold. It warns if everything merged or everything split. Read its output before submitting.

### 7. Spend the reserve (10 answers per endpoint)
`solve` lists the pairs closest to the identity threshold; they are also in the sidecar `.json`. Add answers to the recommended probes on those endpoints only, then run `solve` again:
```bash
python query.py run --target scored --endpoints ep-04,ep-16 --recommended configs/tuned.json --repeats 34 --max-total 100 --yes
```
More answers can only sharpen a pair's evidence: same-model pairs stay near 0, different-model pairs move further away.

## Optional: dedicated family probes (off by default)

To try them, put the probes in `family_probes` in `configs/default.json` (the file has an example list under `family_probes_example`). Then run `python cluster.py tune --budget 80`. `tune` calibrates the family threshold above the largest family distance seen on the references, and prints a second scored collection command (those probes × 2 answers). `solve` then cuts families on those probes.

On the mock, this scored **worse** (0.19–0.26 against 0.725). Coarser families forced the identity feature to separate endpoints across families, which it does poorly. Keep it off unless the real reference data shows otherwise.

## Testing without the real server

The code was tested end to end against a local mock of the API: querying, the "ambiguous → lost" path, idempotent re-runs, tuning, solving, scoring and a submit dry run. `ME_SERVER_URL` points the code at another server.

## Known limits

- The references are a different, smaller model than the scored endpoints. A probe that works on them may work less well on the scored endpoints.
- z-scores depend on the number of answers per probe. Tuning and scored collection must use the same R, which `--recommended` with `--repeats R` takes care of.
- The family split is the weakest part (family MCC about 0.40 on the mock). It is only 10 % of the score.
- The `prefix` feature only means something when an endpoint decodes greedily or at low temperature. `features.py` prints each endpoint's decode mode.
- `tune` writes the identity threshold as a single number. A pair just above or below it is a coin flip: that is what the reserve in step 7 is for.
- The probes are a first draft. Expect to replace some after reading real reference answers.
