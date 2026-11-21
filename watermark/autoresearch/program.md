# autoresearch — watermark localisation

Adapted from karpathy/autoresearch: an agent edits one file, runs a fixed-budget experiment, keeps or discards.

## Setup
1. Run tag `nov21`; branch `autoresearch/nov21-wm` (fresh) from `jacquet2/watermark`.
2. In-scope files: `../baseline.py` (feature extractors, fixed), `prepare.py` (caches, folds, metric, submission
   writer — fixed, do not modify), `train.py` (the only file edited), `worker.py` (GPU job server — fixed).
3. Remote layout (JURECA task root): `work/$USER/ar/baseline.py`, `work/$USER/ar/autoresearch/{prepare,worker}.py`.
   Caches in `work/$USER/ar_cache/` are built by the worker on first start.
4. `results.tsv` (tab-separated, untracked) with header `commit	val_miou	memory_gb	status	description`.
5. One worker job: `sbatch --time=01:30:00 ../gpu_job.sh work/$USER/ar/autoresearch/worker.py` (1 node, 1 GPU).

## Experimentation
- Each experiment: upload `train.py` to `work/$USER/ar_queue/<commit>/train.py`, then create `READY`.
  The worker runs it (cwd = that folder), output in `run.log`, exit code in `DONE`.
- Budget: ~5 min (`TIME_BUDGET = 300`); the worker kills a run after 10 min (= crash/discard).
- Metric: `val_miou` = official two-class per-image mIoU on out-of-fold predictions, 3 folds over all 510 train
  images (stratified method × full_watermark). Higher is better. Read with `grep "^val_miou:" run.log`.
- CAN: change anything in `train.py`: inputs used among cached features, head, loss, optimiser, epochs, batch,
  augmentations that keep features/labels consistent (e.g. joint flips are NOT valid on cached extractor maps
  unless argued), post-processing, threshold strategy.
- CANNOT: modify `prepare.py`, `worker.py`, `baseline.py`, the folds or the metric; install packages; use
  validation-fold masks to fit a fold's model; select epochs per fold on its own validation masks. Thresholds may
  be chosen on the out-of-fold predictions (one value per method: small, documented optimism).
- Simplicity: equal score with less code = keep; tiny gains with ugly complexity are not worth it.
- First run: baseline as is.

## Logging and git
LOOP: commit the `train.py` change → run → read `val_miou` / `peak_vram_mb` → append a row to `results.tsv` →
if `val_miou` improved strictly, keep the commit (branch advances); otherwise `git reset --hard HEAD~1`.
Crash: fix trivial bugs and rerun; otherwise log `crash` (val 0.000000) and move on.

## Final
The best commit's `train.py --final output/$USER/ar_<tag>/submission.pt` fits on all train images and writes a
validated test submission. Uploading to the leaderboard is a separate, human-approved step.
