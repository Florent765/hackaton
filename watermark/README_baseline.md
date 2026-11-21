# Watermark localisation baseline

Frozen organiser extractors (`wmloc/watermarks.py`) + one small segmentation head per method, trained on the
official 128×128 masks and selected on the official two-class per-image mIoU.

## Files

| File | Role |
| --- | --- |
| `baseline.py` | modes `smoke`, `train`, `evaluate`, `predict` |
| `run_all.py` | chains train → evaluate → predict inside one Slurm job (never uploads to the leaderboard) |

## Inputs of the head

| Method | Frozen maps (from the wrapper) | Agreement with known message | Notes |
| --- | --- | --- | --- |
| PixelSeal | `embedding`, `convnext` (8×8 → 32×32 by shift-and-stitch) | yes, from `per_position_logits` | input resized to 256 by the wrapper |
| TrustMark | `layer2` (31×31), `layer3` (16×16) | yes, `per_position_logits` 8×8 → 32×32 | weights from a private copy (see below) |
| StegaStamp | `dec_conv_3` (100×100), `dec_conv_5` (50×50) | no (dense layers, no per-position logits) | copy of the ONNX graph with extra outputs |

All methods also receive the RGB image pooled to 64×64 and two image-level scalars computed from the global
logits and the known message (soft agreement, bit accuracy).

Agreement map: `sum_k (2 m_k − 1) · l_k(h, w) / nbits`, where `l(h, w)` are the decoder's own logits before its
global mean pooling. No bit-to-pixel mapping is assumed.

Loss: BCEWithLogits + per-image two-class soft Dice (eps = 1, so an empty class predicted empty costs ~0).
Selection: best epoch and threshold (grid 0.20–0.80, step 0.05) by validation mIoU, per method.
Split: 20 % validation, seed 0, stratified by method × `full_watermark`. Source-image grouping is impossible:
names are hashes and all messages are distinct.

## One-time setup (already done for jacquet2)

Compute nodes have no internet and the shared `.venv` has no TrustMark weights. Without modifying the shared
environment, copy the package and fetch its weights **on the login node** (download only):

```bash
R=/p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task
cd $R && mkdir -p work/$USER/pylib && cp -r .venv/lib/python3.11/site-packages/trustmark work/$USER/pylib/
cd work/$USER/pylib && $R/.venv/bin/python -c 'import trustmark, pathlib; from trustmark.trustmark import TrustMark as T; d=pathlib.Path(trustmark.__file__).parent/"models"; [T.check_and_download(None, str(d/f)) for f in ["trustmark_Q.yaml","trustmark_rm_Q.yaml","decoder_Q.ckpt","encoder_Q.ckpt","trustmark_rm_Q.ckpt"]]'
```

`baseline.py --pylib work/$USER/pylib` (default) prepends it to `sys.path`. The StegaStamp feature ONNX copy is
created automatically in `work/$USER/models/`.

## Commands

From the laptop, upload to a fresh run folder (never edit files of a queued/running job):

```bash
RUN=baseline_v1
ssh jureca "mkdir -p /p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task/work/$USER_JSC/$RUN"
rsync -av watermark/baseline.py watermark/run_all.py \
  "jureca:/p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task/work/$USER_JSC/$RUN/"
```

On JURECA, always from the task root:

```bash
jutil env activate -p training2666
cd /p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task
mkdir -p output/$USER/baseline_v1
RUN=work/$USER/baseline_v1

# smoke (1 GPU, ~1 min)
sbatch --time=00:20:00 ../gpu_job.sh $RUN/baseline.py smoke --smoke-n 8

# full pipeline: train -> evaluate -> predict (1 node, 1 GPU)
sbatch --time=01:00:00 ../gpu_job.sh $RUN/run_all.py --out output/$USER/baseline_v1

# or stage by stage
sbatch --time=01:00:00 ../gpu_job.sh $RUN/baseline.py train --out output/$USER/baseline_v1
sbatch --time=00:20:00 ../gpu_job.sh $RUN/baseline.py evaluate --ckpt-dir output/$USER/baseline_v1 --out output/$USER/baseline_v1
sbatch --time=00:30:00 ../gpu_job.sh $RUN/baseline.py predict --ckpt-dir output/$USER/baseline_v1 \
       --submission output/$USER/baseline_v1/submission.pt

# training of the 3 methods in parallel on one node (max 1 node, 4 GPUs)
sbatch --gres=gpu:4 --cpus-per-task=32 --time=01:00:00 ../gpu_job.sh $RUN/baseline.py train --parallel \
       --out output/$USER/baseline_v1
```

Outputs in `output/$USER/baseline_v1/`: `head_<method>.pt` (state dict, config, split, model info, threshold,
validation metrics), `split.json`, `history_<method>.json`, `train_summary.json`, `eval_report.json`,
`viz/val_<method>.png` (worst 4 and best 4 validation images: image | ground truth | prediction),
`submission.pt` (reloaded with `weights_only=True` and checked: keys, names, duplicates, shape, binary values,
size ≤ 20 MB).

Submitting to the leaderboard is a separate, manual step (`submission_template.py`, API key kept outside Git).
