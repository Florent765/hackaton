"""autoresearch GPU worker (FIXED). One Slurm job (1 node, 1 GPU) that builds the feature caches if missing, then
runs queued experiments one at a time, so each experiment does not wait in the Slurm queue.

Protocol (queue dir default work/$USER/ar_queue):
  <queue>/<exp_id>/train.py   experiment file, then an empty <queue>/<exp_id>/READY marks it runnable
  worker runs `python train.py [args from ARGS file]` with cwd=<exp_id>, PYTHONPATH=<harness dir>, stdout -> run.log
  then writes <exp_id>/DONE containing the exit code ('timeout' if killed after --timeout seconds)
  <queue>/HEARTBEAT is touched every poll; the worker exits after --idle-exit seconds without work.
Launch from the task root:
  sbatch --time=04:00:00 ../gpu_job.sh work/$USER/ar/autoresearch/worker.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default=str(P.TASK_ROOT / "work" / P.USER / "ar_queue"))
ap.add_argument("--timeout", type=int, default=600)
ap.add_argument("--idle-exit", type=int, default=5400)
a = ap.parse_args()
Q = Path(a.queue); Q.mkdir(parents=True, exist_ok=True)


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


missing = [m for s in ("train", "test") for m in P.METHODS if not (P.CACHE / f"{s}_{m}.pt").exists()]
if missing:
    log("building caches, missing:", missing)
    t0 = time.time(); P.build(); log(f"caches built in {time.time() - t0:.0f}s")
(Q / "WORKER").write_text(f"{os.environ.get('SLURM_JOB_ID')} {os.uname().nodename} started {time.ctime()}\n")
last = time.time()
while True:
    (Q / "HEARTBEAT").write_text(time.ctime())
    todo = sorted((d for d in Q.iterdir() if d.is_dir() and (d / "READY").exists() and not (d / "DONE").exists()),
                  key=lambda d: (d / "READY").stat().st_mtime)
    if not todo:
        if time.time() - last > a.idle_exit:
            log("idle, exiting"); break
        time.sleep(3); continue
    d = todo[0]; last = time.time()
    args = (d / "ARGS").read_text().split() if (d / "ARGS").exists() else []
    log("run", d.name, args)
    env = dict(os.environ, PYTHONPATH=f"{HERE}:{os.environ.get('PYTHONPATH', '')}")
    t0 = time.time()
    with open(d / "run.log", "w") as fo:
        p = subprocess.Popen([sys.executable, "-u", "train.py", *args], cwd=d, env=env, stdout=fo, stderr=subprocess.STDOUT)
        try:
            code = str(p.wait(timeout=a.timeout))
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); code = "timeout"
    (d / "DONE").write_text(code)
    log("done", d.name, code, f"{time.time() - t0:.0f}s")
    last = time.time()
