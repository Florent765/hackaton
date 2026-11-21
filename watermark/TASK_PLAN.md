# Watermark localization: agent execution plan

Status: audited on 2026-11-21; implementation has not started in this checkout.

## Objective and authoritative sources

Predict which regions of an edited image still carry its embedded watermark. This is binary spatial localization, with the watermark method and message already supplied.

- Organizer task PDF: `Watermark Localization.pdf`, local copy at audit time `/Users/florentdelorme/Documents/repositories/hackathon/Watermark Localization.pdf` (4 pages).
- Event deck: `EuroAIChampionshipParis2026-11-21.pdf`, pages 26–32.
- Starter kit: <https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task>. Read actual data loader, decoder wrappers, metrics, and submission template before implementing. Their implementation was not inspected in this audit.
- Background: Pixel Seal, <https://arxiv.org/html/2512.16874> or <https://arxiv.org/pdf/2512.16874>.
- Repository operations: `../TEAM_GUIDE.md` and `../gpu_job.sh`.

Use organizer instructions and the actual dataset/submission contract as authoritative. Record discrepancies with this plan.

## Verified task contract

- Methods: PixelSeal, TrustMark, StegaStamp.
- Training: 510 images, 170 per method, with 512×512 and 128×128 ground-truth masks.
- Test: 1,020 images, 340 per method.
- Both splits supply the method and embedded message. Do not spend effort identifying the method or guessing the bits.
- Input images are 512×512. Output masks are binary 128×128: `1 = watermarked`, `0 = clean`.
- A ground-truth grid cell is watermarked when at least 50% of its 4×4 image pixels are watermarked. Do not substitute bilinear resizing for this rule when constructing training targets.
- Metric: compute clean-class IoU and watermarked-class IoU per image, average the two, then average over images. An empty class union receives IoU 1. Use the exact starter implementation as the reference.
- Submit `.pt` containing `{"image_names": names, "masks": tensor[N,128,128]}`. Every expected image name must occur exactly once; values must be binary; maximum file size 20 MB.
- Use bool/uint8 masks: uint8 payload is about 16.7 MB for the test set, while float32 is about 66.8 MB before serialization overhead. Measure the actual saved file size.
- Public evaluation covers 30% of test images, final evaluation the remaining 70%. The leaderboard retains the team's best public result; current submission score is returned separately.

## Ordered implementation

1. **Inspect the actual assets.** Locate train/test metadata, images, masks, downloaded models, and wrapper interfaces. Verify name ordering, message encoding/bit length per method, normalization, decoder input sizes, and whether soft bit logits or intermediate spatial features are exposed. Do not assume wrapper tensor shapes.
2. **Implement exact validation.** Reproduce official per-image two-class mIoU, including empty-union behavior. Compare against the supplied metric. Establish constant-zero/constant-one references. Visually inspect image/mask alignment.
3. **Create a fixed holdout.** Stratify by watermark method and inspect mask coverage. Keep variants of the same source image together to avoid leakage. Save split IDs and seed. Report overall, per-method, and coverage-stratified mIoU; do not select checkpoints on training IoU.
4. **Build a decoder-based baseline.** Route each image to its known method's extractor. Run overlapping crops at a few scales, score compatibility with the supplied message, project scores back to image coordinates, and blend overlaps. Use soft bit agreement/BCE if available; handle hard outputs according to the actual wrapper. Tune crop size, stride, resizing, and thresholds per method on holdout data.
5. **Cache expensive inference.** Store crop scores/features with model identity, preprocessing, crop coordinates, and configuration hashes. Reuse caches for threshold/combination experiments; benchmark a small batch before scaling to all images.
6. **Train a small segmentation head.** Freeze the supplied extractor initially and train a lightweight head on its intermediate spatial features, optionally incorporating known-message compatibility. Produce 128×128 probabilities. Compare BCE/Dice-style objectives and select thresholds using actual holdout mIoU. With only 170 labeled images per method, avoid initially fine-tuning the entire decoder.
7. **Augment carefully.** Geometric transforms must preserve image/mask alignment. Photometric, compression, resize, and crop operations can change the watermark signal, so validate them rather than assuming labels remain meaningful under arbitrary corruption. Resolve external/synthetic-data policy before relying on additional generated data.
8. **Combine only demonstrated improvements.** Compare feature-head predictions, crop maps, and their combination. Add multi-scale inference or conservative boundary refinement only if gains justify runtime. Preserve separate per-method thresholds/configurations if validated.
9. **Run final inference and export.** Use deterministic metadata-to-output mapping and check every test item. If retraining on all labeled data, fix training duration and thresholds from the held-out experiments first. Preserve the validated candidate as a fallback.

## Important method limits

Initial watermark placement is rectangular, but later image edits can produce holes and transplanted objects. Do not force final masks into rectangles. Pixel accuracy and whole-image bit accuracy do not measure localization quality. Crop resizing can alter watermark spatial frequencies, and known-message agreement is a heuristic score rather than a calibrated pixel probability.

The Pixel Seal paper explains embedding/extraction and robustness training; it does not establish that a proposed crop score localizes edited regions. Reproducing adversarial watermark-embedding training is outside the required prediction deliverable. Validate localization ideas on the released masks.

## Files to implement and completion checks

Suggested files: `inspect_data.py`, `evaluate.py`, `extract_features.py`, `train.py`, `predict.py`, `validate_submission.py`, and small configs. Reuse organizer loaders/wrappers where possible and keep their files unchanged.

Before any submission:

- Reload the saved `.pt` and check required keys, exact test-name coverage, no duplicates, and correct ordering relative to stored masks.
- Check tensor shape `[1020,128,128]`, binary values, supported bool/uint8 dtype, and actual file size against 20 MB.
- Visually inspect predicted overlays for every method and several mask coverages.
- Record validation results, checkpoint/config hashes, threshold choices, and output hash.
- Verify current submission client, authentication, cooldown/quotas, and deadline from actual instructions; these were not established in the audit.

Done means a reproducible inference pipeline, valid binary artifact, measured holdout comparisons by method, and recorded limitations. A decoder producing correct bits is not sufficient proof of localization quality.

## Execution environment and immediate open checks

The remote task root is `/p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task` (the local code folder is `watermark/`). Downloaded models are expected under `models/`, extracted data under `release/`; verify actual layout. Upload to `work/<username>/<run-id>/` and use immutable run directories for queued/running jobs. Submit GPU work from the remote task root with `sbatch ../gpu_job.sh work/<username>/<run-id>/train.py ...` after creating `output/`. The launcher defaults to 30 minutes. Do not recreate shared environments.

Start with one GPU for decoder inference and the small head, then benchmark. Resolve feature access, source-image grouping, synthetic/external-data rules, and submission quotas before depending on them. Never print or commit credentials, data, model weights, or generated outputs. The event deck states the competition ends **2026-11-22 at 12:00 local event time**; leave a final submission buffer.
