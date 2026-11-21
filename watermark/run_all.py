"""Chain baseline.py train -> evaluate -> predict in one Slurm job (the shared gpu_job.sh runs one Python script).
Run from the task root:  sbatch --time=01:00:00 ../gpu_job.sh work/<user>/<run>/run_all.py --out output/<user>/<run>
Extra arguments after --out are passed to every stage (e.g. --epochs 40). Stops at the first failing stage.
Never uploads anything to the leaderboard."""
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
args = sys.argv[1:]
out = args[args.index("--out") + 1] if "--out" in args else "output/baseline"
rest = [a for i, a in enumerate(args) if a != "--out" and (i == 0 or args[i - 1] != "--out")]
stages = [
    ["train", "--out", out],
    ["evaluate", "--ckpt-dir", out, "--out", out],
    ["predict", "--ckpt-dir", out, "--submission", f"{out}/submission.pt"],
]
for st in stages:
    cmd = [sys.executable, "-u", str(here / "baseline.py"), *st, *rest]
    print(time.strftime("%H:%M:%S"), "RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    code = subprocess.call(cmd)
    print(time.strftime("%H:%M:%S"), f"stage {st[0]} exit {code} in {time.time() - t0:.0f}s", flush=True)
    if code:
        sys.exit(code)
print("ALL STAGES DONE", flush=True)
