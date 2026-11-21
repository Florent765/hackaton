"""Start several autoresearch workers inside ONE Slurm job (CLAUDE.md: one job per team, parallelise inside it).
One worker per visible GPU times --per-gpu. Launch from the task root, chained after the previous team job:
  sbatch --gres=gpu:4 --cpus-per-task=32 --time=01:30:00 --dependency=afterany:<jobid> \
         ../gpu_job.sh work/$USER/ar/autoresearch/multi_worker.py --per-gpu 2"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--per-gpu", type=int, default=2)
ap.add_argument("--idle-exit", default="1800")
a = ap.parse_args()
here = Path(__file__).resolve().parent
visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v] or \
          [str(i) for i in range(torch.cuda.device_count())]
logdir = Path(os.environ.get("AR_LOGDIR", Path("output") / os.environ.get("USER", "user") / "ar"))
logdir.mkdir(parents=True, exist_ok=True)
jid = os.environ.get("SLURM_JOB_ID", "x")
procs = []
for g in visible:
    for k in range(a.per_gpu):
        log = open(logdir / f"worker_{jid}_gpu{g}_{k}.log", "w")
        procs.append(subprocess.Popen([sys.executable, "-u", str(here / "worker.py"), "--idle-exit", a.idle_exit],
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=g), stdout=log, stderr=subprocess.STDOUT))
print(f"started {len(procs)} workers on GPUs {visible}", flush=True)
sys.exit(max(p.wait() for p in procs))
