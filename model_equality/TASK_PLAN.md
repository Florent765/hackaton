# Model equality testing: agent execution plan

Status: audited on 2026-11-21; implementation has not started in this checkout.

## Objective and authoritative sources

Assign each of 33 scored endpoints (`ep-01` through `ep-33`) a base-model family and an exact model identity.

- Organizer task PDF: `Model Equality Testing.pdf`, local copy at audit time `/Users/florentdelorme/Documents/repositories/hackathon/Model Equality Testing.pdf` (4 pages).
- Event deck: `EuroAIChampionshipParis2026-11-21.pdf`, pages 34–41.
- Starter kit: <https://huggingface.co/datasets/SprintML/model_equality>. Inspect actual instructions, API client, `practice_answers.csv`, `score_practice.py`, and `submission.py`. Starter implementation was not inspected in this audit.
- Background paper: <https://arxiv.org/pdf/2410.20247>, especially sections 2–4.
- Repository operations: `../TEAM_GUIDE.md`.

Organizer task instructions and current API behavior take precedence over recommendations below. Record discrepancies explicitly.

## Identity definition: critical distinction

| Endpoint change | Changes model identity? |
| --- | --- |
| Base checkpoint | Yes |
| Quantization configuration | Yes |
| LoRA adapter | Yes |
| Watermark | No |
| System prompt | No |
| Decoding strategy | No |

Family means the same base model. Model identity means checkpoint plus quantization plus LoRA configuration. Output-distribution inequality is therefore not equivalent to different identity.

The paper's MMD tests can provide response-comparison features, but may detect nuisance differences that the task explicitly treats as the same model. Calibrate on practice examples rather than using a p-value directly as the identity label. Failure to reject distribution inequality also does not prove identity.

## Verified budget, score, and submission contract

- Maximum **100 generated responses per scored endpoint across the entire competition**. Multiple responses in one request each consume budget.
- Maximum 256 generated tokens per response and 1,024 input tokens per prompt.
- Maximum **four concurrent requests per team**, across all processes and teammates.
- Responses contain generated text only. Temperature and seed settings are ignored; system messages are rejected. Invalid requests are not counted. Network timeouts do not establish whether a valid generation was consumed.
- Nine practice endpoints (`ref-01` through `ref-09`) have published identities. Local scoring with `score_practice.py` is unlimited; this does not establish unlimited practice API calls. Do not assume practice models share scored-endpoint identities.
- Family score: MCC over all 528 unordered endpoint pairs. Model score: MCC over pairs within a true family. Each MCC is clipped below at zero.
- Final task score: `0.1 * MCC_family + 0.9 * MCC_model`.
- CSV columns: `endpoint,family,model`, exactly one row per endpoint. Labels are arbitrary, but every shared model label must belong to one shared family label.
- Submission cooldown is **30 minutes, including failed submissions**. Public scoring covers only part of the pairs; final evaluation uses unseen pairs.

## Ordered implementation

1. **Inventory before querying.** Read starter code, practice labels, API metadata, and current quota. Find out whether teammates have already consumed scored responses. Establish one owner for all scored queries.
2. **Implement the shared query ledger first.** Persist endpoint, prompt ID and exact text, request parameters, raw response, timestamp, status, and budget accounting. Serialize ledger updates, cap concurrency globally at four, and reserve budget before dispatch. Reconcile ambiguous timeouts; do not blindly retry. Check for reusable cached responses before every request.
3. **Build local scoring and CSV validation.** Reproduce practice scoring and verify family/model nesting. Inspect the official scorer's handling of degenerate MCC denominators instead of guessing. Do this without scored API calls.
4. **Develop practice probes.** Use tightly specified short outputs across factual continuations, code completion, arithmetic, multilingual text, and structured choices. Repeat identical prompts to estimate sampling variation. Avoid relying on self-reported model names or stylistic differences alone.
5. **Compare evidence on known practice identities.** Evaluate positional character/token agreement, string/Hamming-kernel MMD, lexical/style features, and correctness/choice patterns. Measure whether each feature separates changed quantization/LoRA from same-model watermark/system/decoding disguises. Keep some practice probes out of tuning where feasible.
6. **Collect an initial scored batch only after practice calibration.** A provisional allocation is 8 common prompts × 5 repetitions = 40 responses per endpoint, leaving 60 for follow-up. Revise this allocation using practice results and actual remaining quota. Apply the same initial probes across endpoints so comparisons are meaningful.
7. **Cluster consistently.** First infer family groups, then identity subgroups. Preserve pairwise evidence and uncertainty. Use a clustering method that produces transitive groups; do not emit contradictory independent pair labels. Compare candidate partitions using practice MCC.
8. **Spend the reserve adaptively.** Focus repeated probes on ambiguous within-family merges/splits because that component carries 90% of the score. Document the expected information gain before each new batch. Do not exhaust all endpoints automatically.
9. **Export and submit validated candidates.** Preserve every submitted partition and rationale. Use new response evidence and practice calibration to revise clusters, rather than tuning solely to the partial public score.

## Files to implement and completion checks

Suggested files: `query.py`, `probes.json`, `features.py`, `cluster.py`, `validate_submission.py`, and experiment configs. Keep raw responses and ledgers outside Git in the shared task output area, accessible to the team; keep API keys separate.

Before a scored collection:

- Confirm endpoint-specific remaining quota and team-wide concurrency ownership.
- Confirm prompt/input/output limits and ledger persistence.
- Ensure retries cannot silently duplicate requests or exceed the budget.

Before submission:

- Exact header and all 33 expected endpoint IDs, each exactly once.
- Nonempty family/model labels and consistent model-to-family nesting.
- Artifact/config hash and the evidence version used to construct the partition.
- Current cooldown checked; validate before sending because failures also trigger cooldown.

Done means a reproducible partition, valid CSV, auditable response evidence, budget ledger, and documented ambiguous groups. A statistically significant text difference alone is not a completed identity decision.

## Execution environment and immediate open checks

The installed task root is `/p/scratch/training2666/delorme2/team51/model_equality`. Follow `../TEAM_GUIDE.md` for isolated uploads and shared environments. Initial API collection and analysis need little or no GPU compute.

Resolve API URL/authentication, practice API quotas, existing scored usage, timeout/retry semantics, and current announcements before collection. Never print or commit credentials. The event deck states the competition ends **2026-11-22 at 12:00 local event time**; leave time for cooldown and submission recovery.
