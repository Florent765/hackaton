# Model equality testing: execution plan v2

Status: written 2026-11-21. Supersedes the method sections of `TASK_PLAN.md`; the task facts, budget, score and submission contract below are copied from v1 unchanged. A first implementation is in this folder, see [README.md](README.md).

**Update after reading the starter kit (2026-11-21):** the reference endpoints have **no query limit** (answers open question 1). All nine are **one family** with model groups S = {ref-01 plain, ref-03 system prompt, ref-05 persona, ref-06 watermark, ref-08 sampling T=1.0}, T1 = ref-02 (fp8), T2/T3/T4 = ref-09/04/07 (LoRA 0.3/0.7/1.5). Step 6 therefore calibrates the model split only; the family split cannot be tuned on references. Answers are sampled by default, so purely greedy endpoints may be rare.

What changes from v1, in short:

1. Probes are **forced-choice and random-sequence prompts** designed to expose token-preference bias, not generic tasks.
2. Each response carries **many samples** (e.g. 100+ bits), so the 100-response budget yields thousands of observations per endpoint.
3. The **decoding mode of every endpoint is detected first**, and comparisons are made only between compatible modes.
4. A **greedy-divergence feature** targets the within-family split (quantization / LoRA), which carries 90% of the score and is where MMD is known to be weak.
5. Clustering thresholds are **tuned to maximise MCC on practice endpoints**, and the reserve budget goes to pairs near the threshold.

## 1. Objective and sources

Assign each of 33 scored endpoints (`ep-01` … `ep-33`) a base-model family and an exact model identity.

- Organizer task PDF `Model Equality Testing.pdf` (4 pages) and event deck pages 34–41.
- Starter kit: <https://huggingface.co/datasets/SprintML/model_equality> (API client, `practice_answers.csv`, `score_practice.py`, `submission.py`). Not yet inspected.
- Organizer instructions and live API behaviour take precedence over this plan. Record any discrepancy here.

Background reading, by relevance to this plan:

| Paper | What we take from it |
|---|---|
| Gao, Liang, Guestrin, *Model Equality Testing* ([2410.20247](https://arxiv.org/abs/2410.20247)) | Two-sample framing; Hamming-kernel MMD as a baseline feature. |
| Cai et al., *Are You Getting What You Pay For?* ([2504.04715](https://arxiv.org/pdf/2504.04715)) | MMD-style tests have low power against quantization. Do not rely on MMD for the identity split. |
| Zhu et al., *Rank-Based Uniformity Test* (RUT, ICLR 2026, [2506.06975](https://arxiv.org/abs/2506.06975)) | One API call per prompt, beats MMD on quantization, needs reference samples from a known model. |
| LLMPrint ([2509.25448](https://arxiv.org/pdf/2509.25448)) | First-token forced-choice preferences separate ~700 post-trained / quantized variants. |
| FLIPS ([2606.03330](https://arxiv.org/html/2606.03330)) | Bias in generated random binary sequences; instance-level identification with few queries. |
| *One Token Is Enough* ([2607.10252](https://arxiv.org/abs/2607.10252)) | Distribution of single-token answers ("random number") as a fingerprint. |
| LLMmap ([2407.15847](https://arxiv.org/html/2407.15847v3)) | Crafted queries robust to system prompts and sampling settings. |

All reported accuracies are the authors' own benchmarks, not independent replications. None of these papers tests LoRA substitution directly (closest: RUT's SFT setting).

## 2. Identity definition

| Endpoint change | Changes model identity? |
| --- | --- |
| Base checkpoint | Yes |
| Quantization configuration | Yes |
| LoRA adapter | Yes |
| Watermark | No |
| System prompt | No |
| Decoding strategy | No |

Family = same base model. Identity = checkpoint + quantization + LoRA. Output-distribution inequality is therefore **not** identity inequality: watermark, system prompt and decoding all change outputs without changing identity. Every feature must be validated on practice endpoints for whether it reacts to these nuisances.

## 3. Budget, score and submission contract (verified in v1)

- Max **100 generated responses per scored endpoint for the whole competition**. Several responses in one request each count.
- Max 256 generated tokens per response, 1,024 input tokens per prompt.
- Max **4 concurrent requests per team**, across all processes and teammates.
- Responses contain generated text only. Temperature and seed are ignored; system messages are rejected. Invalid requests are not counted. A network timeout does not tell whether a generation was consumed.
- Practice endpoints `ref-01` … `ref-09` have published identities. `score_practice.py` is unlimited locally; practice API quota is unknown. Practice identities need not overlap scored identities.
- Family score: MCC over all 528 endpoint pairs. Model score: MCC over pairs within a true family. Each clipped at 0.
- **Final = 0.1 × MCC_family + 0.9 × MCC_model.**
- CSV `endpoint,family,model`, one row per endpoint; a model label must map to exactly one family label.
- **Submission cooldown 30 min, failed submissions included.** Public score uses part of the pairs; final uses unseen pairs.
- Competition ends **2026-11-22 12:00 local event time**.

## 4. Ordered implementation

### Step 0. Inventory (no scored calls)
Read starter code, `practice_answers.csv`, API metadata and remaining quota. Ask the team whether anyone has already spent scored responses. Name **one owner** for all scored queries.

### Step 1. Query ledger (before any call)
Same requirements as v1: persist endpoint, prompt ID, exact text, parameters, raw response, timestamp, status; reserve budget before dispatch; global concurrency cap 4; serialized writes; no blind retries; cache lookup before every request. Add a `probe_family` column (`decode`, `choice`, `bits`, `numbers`, `greedy_long`, `reserve`) so spend per probe type is visible.

### Step 2. Local scorer and CSV validator
Reproduce `score_practice.py`, check family/model nesting, and read how the official scorer handles a degenerate MCC denominator (e.g. all singletons).

### Step 3. Probe bank (`probes.json`)
Four probe types. Every prompt must ask for a strictly formatted output so that parsing is trivial and refusals or chatter are detectable.

| Type | Example prompt (shape only) | Samples per response | Target |
|---|---|---|---|
| `decode` | A fixed open-ended prompt, sent 3× | n/a | Detect greedy vs sampling (§ Step 4). |
| `choice` | "Answer with exactly one word: heads or tails." / "Pick one: A or B." / "One digit 0–9." | 1 | First-token preference (LLMPrint, One Token). Family and identity. |
| `bits` / `numbers` | "Output 200 random characters, each 0 or 1, no spaces." / "List 80 random integers from 1 to 10, comma-separated." | 80–200 | Sequence bias: symbol frequencies, run lengths, bigram transitions (FLIPS). Most samples per unit of budget. |
| `greedy_long` | A fixed neutral continuation task that fills ~256 tokens (e.g. "Continue this text: …") | 1 long sequence | First-divergence position between endpoints (§ Step 5). Targets quantization / LoRA. |

Write 3–5 variants of each type with different wording, so that a single system prompt cannot dominate one probe. Keep one variant per type **held out** from tuning.

### Step 4. Detect decoding mode
Send the `decode` prompt 3× to each endpoint (3 responses).
- 3 identical outputs → **greedy** (or near-greedy).
- Otherwise → **sampling**.

Comparison rules that follow:
- greedy vs greedy: compare outputs directly; divergence position is informative.
- sampling vs sampling: compare distributions (frequencies, MMD, RUT-style ranks).
- greedy vs sampling: compare the greedy output with the **mode** of the sampled answers on `choice` probes. Do not compare raw distributions; that only detects the decoding difference, which does not change identity.

### Step 5. Features (`features.py`)
Per endpoint pair, compute:

1. **Choice agreement**: for each `choice` probe, distance between answer distributions (total variation or Jensen–Shannon); for greedy endpoints, argmax agreement.
2. **Sequence bias**: from `bits` / `numbers`, the vector of symbol frequencies, bigram transition matrix and run-length histogram; distance between endpoints, with a bootstrap confidence interval.
3. **Greedy first-divergence index** (greedy endpoints only): token or character position where two outputs on the same `greedy_long` prompt first differ, normalised by length. Hypothesis: same identity → identical or diverging only through a system prompt or watermark; quantized or LoRA variant → long shared prefix, then divergence; different family → early divergence. **Must be confirmed on practice endpoints** before use.
4. **Hamming-kernel MMD** on short outputs, as in Gao et al. Use it as a family-level feature, not as the identity decision.
5. **Optional RUT**: if a family is identified with a downloadable open-weight checkpoint and a teammate has GPU access, generate reference samples locally and run the rank-based uniformity test against each endpoint of that family. This costs no extra API budget beyond responses already collected.

**Watermark trade-off.** A watermark adds a fixed logit bias to a pseudo-random token subset. On near-tie probes that bias can exceed the quantization shift, so a watermarked copy may look like a different identity. Low-entropy probes are robust to the watermark but blind to quantization. Use both, and on practice endpoints measure which features keep a watermarked endpoint with its unwatermarked twin.

### Step 6. Calibrate on practice endpoints
Run the whole probe bank on `ref-01` … `ref-09` (within the practice quota, which must be checked first). For each feature, measure on known pairs:
- same identity under nuisance (watermark / system prompt / decoding) vs different identity in the same family;
- same family vs different family.

Keep the features that separate these on the held-out variants. Fit pair-similarity weights or thresholds; prefer a simple model (logistic regression on 4–6 features) given only 36 practice pairs.

### Step 7. First scored batch (≈40 responses per endpoint)
Identical probes on all 33 endpoints:

| Probe | Responses |
|---|---|
| `decode` | 3 |
| `choice` (several prompts × repeats) | 12 |
| `bits` / `numbers` | 13 |
| `greedy_long` | 12 |
| **Total** | **40** |

Revise this split from practice results before sending anything. Leave 60 per endpoint in reserve.

### Step 8. Cluster (`cluster.py`)
1. **Family pass:** agglomerative (average linkage) or correlation clustering on family-level similarity; threshold tuned to maximise family MCC on practice.
2. **Identity pass:** within each family only, same procedure on identity-level features, threshold tuned for model MCC.
3. Output is a partition, so labels are transitive by construction. Keep pairwise scores and confidence for Step 9.

MCC note: when unsure, check on practice whether over-merging or over-splitting costs more with the actual family sizes, and bias the threshold accordingly.

### Step 9. Spend the reserve adaptively
List the pairs whose similarity is closest to the identity threshold. For each, write down the expected effect before spending: which probes, how many responses, and which decision it could flip. Spend in small batches (≤10 responses per endpoint), re-cluster, repeat. Never spend all 100 on an endpoint automatically.

### Step 10. Submit
Validate with the local validator first (failures also trigger the 30-min cooldown). Store each submitted CSV with its feature-config hash, ledger snapshot and a one-line rationale. Use the public score as one signal among others, not as the tuning target. Plan the last submission **before 11:00 on 2026-11-22** to leave room for one cooldown and a fix.

## 5. Files

| File | Content |
|---|---|
| `query.py` | API client + ledger, concurrency cap, budget reservation. |
| `probes.json` | Probe bank with type, variant ID, held-out flag. |
| `features.py` | Parsers and pairwise features (§ Step 5). |
| `cluster.py` | Two-pass clustering, threshold tuning on practice. |
| `validate_submission.py` | Header, 33 IDs, nesting, non-empty labels. |
| `configs/` | Feature weights and thresholds per submission. |

Raw responses and ledgers stay outside Git in the shared task output folder. API keys stay out of files.

## 6. Checks

Before any scored collection:
- Endpoint-level remaining quota confirmed; one person owns scored querying.
- Ledger persists and cannot double-spend on retry.
- Probe parsers tested on practice outputs (refusals and malformed outputs handled).

Before each submission:
- Exact header; all 33 endpoints exactly once; family/model nesting holds.
- Config hash and ledger version recorded.
- Cooldown checked.

Done means a reproducible partition, a valid CSV, an auditable ledger, and a short list of ambiguous groups with the evidence for each.

## 7. Environment and open questions

Task root on JURECA: `/p/scratch/training2666/delorme2/team51/model_equality` (see `../TEAM_GUIDE.md`). Collection and analysis need no GPU; only the optional RUT step does.

Open questions for the organizers or the starter kit:
1. Practice-endpoint quota, and whether it is separate from the scored budget.
2. Timeout semantics: is a timed-out generation charged?
3. Do endpoints with the same identity also share a decoding strategy, or is decoding varied independently?
4. Are any scored identities also present among the practice endpoints?
