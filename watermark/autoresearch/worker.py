"""autoresearch GPU worker (FIXED). One Slurm job (1 node, 1 GPU) that builds the feature caches if missing, then
runs queued experiments one at a time, so each experiment does not wait in the Slurm queue.

Protocol (queue dir default work/$USER/ar_queue):
  <queue>/<exp_id>/train.py   experiment file, then an empty <queue>/<exp_id>/READY marks it runnable
  worker runs `python train.py [args from ARGS file]` with cwd=<exp_id>, PYTHONPATH=<harness dir>, stdout -> run.log
  then writes <exp_id>/DONE containing the exit code ('timeout' if killed after --timeout seconds)
  Several workers (separate 1-GPU jobs) can share one queue: experiments are claimed atomically (<exp_id>/CLAIM),
  caches are built by one worker under BUILD.lock while the others wait.
  <queue>/HEARTBEAT_<jobid> is touched every poll; a worker exits after --idle-exit seconds without work.
Launch from the task root:
  sbatch --time=01:30:00 ../gpu_job.sh work/$USER/ar/autoresearch/worker.py
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


def claim(path: Path) -> bool:
    """Atomic claim shared by several workers (O_EXCL create)."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, JOB.encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


JOB = f"{os.environ.get('SLURM_JOB_ID')}@{os.uname().nodename}"
cache_ok = lambda: all((P.CACHE / f"{s}_{m}.pt").exists() for s in ("train", "test") for m in P.METHODS)
if not cache_ok():
    P.CACHE.mkdir(parents=True, exist_ok=True)
    lock = P.CACHE / "BUILD.lock"
    if claim(lock):
        try:
            log("building caches"); t0 = time.time(); P.build(); log(f"caches built in {time.time() - t0:.0f}s")
        finally:
            lock.unlink(missing_ok=True)
    else:
        log("another worker builds the caches, waiting")
        while not cache_ok():
            if not lock.exists() and not cache_ok():             # builder died: take over
                if claim(lock):
                    try:
                        P.build()
                    finally:
                        lock.unlink(missing_ok=True)
            time.sleep(10)
        log("caches ready")
(Q / f"WORKER_{os.environ.get('SLURM_JOB_ID')}").write_text(f"{JOB} started {time.ctime()}\n")
last = time.time()
while True:
    (Q / f"HEARTBEAT_{os.environ.get('SLURM_JOB_ID')}").write_text(time.ctime())
    todo = sorted((d for d in Q.iterdir() if d.is_dir() and (d / "READY").exists() and not (d / "DONE").exists()
                   and not (d / "CLAIM").exists()), key=lambda d: (d / "READY").stat().st_mtime)
    todo = [d for d in todo[:1] if claim(d / "CLAIM")]
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
