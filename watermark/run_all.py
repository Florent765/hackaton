"""Single-node team job (CLAUDE.md: one job per team, a whole node = 4 GPUs, parallelise inside it).

With N >= 2 visible GPUs:
  GPU 0      baseline.py train -> evaluate -> predict, then becomes an autoresearch worker
  GPU 1..N-1 autoresearch workers (autoresearch/worker.py) sharing one experiment queue
With 1 GPU: only the baseline chain.
Run from the task root:  sbatch --gres=gpu:4 ../gpu_job.sh work/<user>/<run>/run_all.py --out output/<user>/<run>
Extra arguments after --out go to every baseline stage. Never uploads anything to the leaderboard.
Worker location: AR_WORKER (default work/$USER/ar/autoresearch/worker.py); idle exit AR_IDLE (default 1200 s) so the
node is released when the experiment queue stays empty."""
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
args = sys.argv[1:]
out = args[args.index("--out") + 1] if "--out" in args else "output/baseline"
rest = [a for i, a in enumerate(args) if a != "--out" and (i == 0 or args[i - 1] != "--out")]
user = os.environ.get("USER", "user")
worker = os.environ.get("AR_WORKER", f"work/{user}/ar/autoresearch/worker.py")
idle = os.environ.get("AR_IDLE", "1200")
ngpu = torch.cuda.device_count()
Path(out).mkdir(parents=True, exist_ok=True)


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


VISIBLE = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v] or [str(i) for i in range(ngpu)]


def gpu_env(i):
    """i indexes the GPUs this process was given (e.g. CUDA_VISIBLE_DEVICES=1,2,3 via srun --overlap)."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=VISIBLE[i])


def start_worker(i):
    f = open(Path(out) / f"worker_gpu{i}.log", "w")
    log(f"start autoresearch worker on GPU {i} -> {f.name}")
    return subprocess.Popen([sys.executable, "-u", worker, "--idle-exit", idle], env=gpu_env(i),
                            stdout=f, stderr=subprocess.STDOUT), f


def baseline_chain(env):
    stages = [["train", "--out", out], ["evaluate", "--ckpt-dir", out, "--out", out],
              ["predict", "--ckpt-dir", out, "--submission", f"{out}/submission.pt"]]
    for st in stages:
        cmd = [sys.executable, "-u", str(here / "baseline.py"), *st, *rest]
        log("RUN", " ".join(cmd)); t0 = time.time()
        code = subprocess.call(cmd, env=env)
        log(f"stage {st[0]} exit {code} in {time.time() - t0:.0f}s")
        if code:
            return code
    return 0


log(f"visible GPUs: {ngpu}")
workers = []
if ngpu >= 2 and Path(worker).exists():
    workers = [start_worker(i) for i in range(1, ngpu)]
code = baseline_chain(gpu_env(0) if ngpu >= 2 else dict(os.environ))
log(f"baseline chain finished with code {code}")
if ngpu >= 2 and Path(worker).exists():
    workers.append(start_worker(0))
for p, f in workers:
    p.wait(); f.close()
log("ALL DONE")
sys.exit(code)
