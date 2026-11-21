# Unlearning task: agent execution plan

Status: audited on 2026-11-21; implementation has not started in this checkout.

## Objective and authoritative sources

Modify the supplied image classifier so its behavior on the forget set approaches models retrained without that set, while preserving behavior elsewhere. Low accuracy on forget images is not the objective by itself.

Read these before implementing:

- Organizer task PDF: `Unlearning.pdf`. The local copy at audit time is `/Users/florentdelorme/Documents/repositories/hackathon/Unlearning.pdf` (5 pages).
- Event deck: `EuroAIChampionshipParis2026-11-21.pdf`, especially pages 43–47.
- Actual starter kit: <https://huggingface.co/datasets/SprintML/unlearning>. Read `task_template.py`, `submission.py`, and supplied instructions from the installed task directory. Their implementation was not inspected in this audit.
- Background: SCRUB, <https://arxiv.org/pdf/2302.09880>, especially sections 3.1–3.2.
- Repository operational instructions: `../TEAM_GUIDE.md` and `../gpu_job.sh`.

Treat organizer instructions and the actual loader/evaluator contract as authoritative. Record any discrepancy with this plan before relying on an assumption.

## Verified task contract

- Model: `timm` `vit_base_patch16_224`, 100 classes, LayerScale initialization `1e-4`. Use the exact starter builder and preprocessing.
- Inputs: `model.pt` and `forget_images.npz`; NPZ key `images`, shape `[3500,224,224,3]`, dtype `uint8`.
- Forget labels, retain training data, and gold-model predictions are not provided.
- The method is expected to work from the original model and forget set alone. Do not make external data a dependency without resolving the organizer policy.
- Submit an original-architecture weights-only state dictionary as `.pt` or `.pth`, maximum 450 MB. Floating weights must be finite float32 values; strict loading must succeed. Merge any adapters before export.
- Server inference uses evaluation mode and fp16 autocast.
- Approximately 30% of evaluation is public and 70% is final; altered copies follow their source image's split. The server retains the best public result and also reports the current submission's result. No submission cooldown or quota was specified in this task PDF; inspect current rules before scheduling evaluations.

## Score: what to optimize

Held-out accuracy below 70% makes the score zero. Otherwise:

```text
final_score = forget_score * retain_score
forget_score = loss_distribution_similarity * prediction_similarity
retain_score = accuracy_ratio * original_prediction_similarity
```

- Loss similarity compares the mean and standard deviation of forget losses against the gold retraining reference: `1 - clip(d_sub / d_orig, 0, 1)`, with `d = abs(mean_loss - gold_mean) + abs(std_loss - gold_std)`.
- Prediction similarity compares each forget image against gold using `KL(gold || submission)` with softmax temperature 2, normalized relative to original-model divergence.
- Forget evaluation includes exact images and hidden altered copies with equal weight.
- Retain accuracy is `clip(submission_accuracy / original_accuracy, 0, 1)` on a hidden control set.
- Retain prediction similarity is `1 / (1 + mean KL(original || submission))`.

The true score cannot be computed locally with the released inputs. Local pseudo-label loss, entropy, and weight drift are diagnostics, not evidence that gold behavior or the 70% gate improved.

## Ordered implementation

1. **Inventory and reproduce the starter.** Locate installed files, inspect requirements and builder, load original weights strictly, and reproduce the prediction helper. Check image range, channel order, normalization, resize behavior, and evaluation mode.
2. **Establish export correctness.** Save the unchanged model through the intended export path, reload with the exact starter architecture, and compare predictions. This establishes a valid artifact, not a competitive unlearning baseline.
3. **Cache reference behavior.** Cache original logits on forget images and mild augmented views, with preprocessing/version metadata. Original argmax predictions may be used as pseudo-labels; never describe them as true labels. Augmented views are not a retain set.
4. **Build a short-run experiment harness.** Every run starts from original weights. Record seed, update scope, optimizer, strength, steps, regularization, augmentation, runtime, and checkpoint hash. Save intermediate checkpoints.
5. **Run conservative pilots.** Compare small pseudo-label gradient-ascent updates to the classifier head and last transformer block, anchored toward original weights. Use tight step limits and finite-value checks. Optionally test mild head-weight/bias scaling as a cheap calibration diagnostic. These are hypotheses, not established solutions.
6. **Control update strength.** Interpolate promising updates as `theta = theta_original + alpha * (theta_candidate - theta_original)` for a small declared set of `alpha` values. Compare local drift and any available server score components. Reject directions that damage utility or repeatedly score zero.
7. **Only expand after evidence.** Try sparse saliency-selected updates or additional layers only if initial results justify them. Keep one GPU initially and benchmark actual throughput before requesting more resources.
8. **Select and export.** Preserve the original checkpoint, the best evaluated candidate, and their configurations separately. Do not choose the last run merely because it is newest.

## Important method limits

Stock SCRUB needs retain examples, supervised/distillation recovery, and checkpoint-selection information absent here. Do not transplant it unchanged or substitute forget augmentations while claiming equivalent recovery. Aggressive ascent, random targets, and entropy maximization can harm the hidden utility gate. Weight anchoring limits drift but does not guarantee retained accuracy.

## Files to implement and completion checks

Suggested files: `inspect_data.py`, `train.py`, `export.py`, `validate_submission.py`, and small run configs. Keep organizer files unchanged and use explicit data/output paths.

Before any submission:

- Load the exported file with the starter's supported weights-only loader and `strict=True`.
- Check all expected keys, shapes, floating dtypes, finite values, and file size against the 450 MB cap.
- Run a small inference batch with server-like eval/autocast settings; verify finite outputs.
- Record artifact SHA-256, config, original-checkpoint identity, and any server result components.
- Verify current authentication, submission limits, and deadline in the actual starter/organizer instructions.

Done means a reproducible candidate, valid reloadable artifact, documented experiment comparison, and clear separation of measured results from unavailable hidden metrics. Do not report a successful training job as a demonstrated unlearning improvement.

## Execution environment and immediate open checks

Run from `/p/scratch/training2666/delorme2/team51/unlearning` on JURECA. Keep uploads in `work/<username>/<run-id>/`; use immutable run directories for queued/running jobs. Submit GPU work with `sbatch ../gpu_job.sh work/<username>/<run-id>/train.py ...` after creating `output/`. The launcher defaults to 30 minutes; override its time limit deliberately if required. Do not recreate shared environments.

First resolve actual API/authentication, existing team submissions, data-use policy, and score components returned by the server. Never log credentials or commit models/data/outputs. The event deck states the competition ends **2026-11-22 at 12:00 local event time**; confirm announcements and finish with a submission buffer.
