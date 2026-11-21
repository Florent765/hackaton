# Team51 hackathon workflow

We share code through GitHub and run experiments on JURECA. Each person uses their own Jülich account, their own local Git clone, and their own branch. The datasets and Python environments are already prepared in our shared JURECA folder.

**Daily workflow:** pull code → edit locally → upload your code to JURECA → submit a GPU job → inspect results → commit and push useful changes.

## Team details

| Item | Value |
| --- | --- |
| GitHub repository | https://github.com/Florent765/hackaton |
| Project | `training2666` |
| Team lead | `delorme2` |
| Teammates | `jacquet2`, `brett2`, `mondoloni1` |
| Shared folder | `/p/scratch/training2666/delorme2/team51` |
| GPU partition | `dc-gpu` |
| Hackathon reservation | `cispahack` |

The reservation was verified active during setup on 21 November 2026, ending on 22 November at 14:00 in the scheduler's displayed time. Check with the organizers if working beyond that window.

## Understand where commands run

| Location | What we do there |
| --- | --- |
| Laptop terminal and editor | Clone GitHub, write code, commit, push, upload files, download results |
| JURECA login shell after SSH | Activate the project, inspect files, submit jobs, read logs |
| GPU compute node | Slurm runs the submitted Python program here |

Pushing to GitHub does not upload code to JURECA. Uploading code to JURECA does not commit it to GitHub. Shared paths on JURECA refer to the same files for all four teammates.

## Set up SSH once per person

### Register your public key

**On your laptop:** confirm in [JuDoor](https://judoor.fz-juelich.de/) that your account has JURECA access through `training2666`.

If you already have a passphrase protected Ed25519 key, you can use it. Otherwise, create one:

```bash
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519_jureca
```

Choose a strong passphrase. Do not overwrite an existing key. Display the public part:

```bash
cat ~/.ssh/id_ed25519_jureca.pub
```

In JuDoor, open **JURECA → Manage SSH-keys**. Upload your public key with a `from=` restriction covering your current public IP addresses or an appropriate allowed range. JuDoor displays your current addresses. The single line has this form:

```text
from="YOUR_PUBLIC_IP_OR_ALLOWED_RANGE" ssh-ed25519 YOUR_PUBLIC_KEY_DATA your-comment
```

Use your own key and addresses. Leave “Remove all other existing public keys” unchecked. Activation may take up to 15 minutes. Upload only the `.pub` content; keep the private key on your laptop.

### Configure a convenient SSH alias

**On your laptop:** add this block to `~/.ssh/config`, preserving any existing entries. Replace `YOUR_JUELICH_USERNAME` with your own username, and adjust `IdentityFile` if your key has a different name.

```sshconfig
Host jureca
    HostName jureca.fz-juelich.de
    User YOUR_JUELICH_USERNAME
    IdentityFile ~/.ssh/id_ed25519_jureca
    IdentitiesOnly yes
    ServerAliveInterval 30
    ControlMaster auto
    ControlPath ~/.ssh/jureca-%C
    ControlPersist 30m
```

`delorme2` currently uses `~/.ssh/id_ed25519`; the other accounts should select their own key. Connection sharing lets subsequent SSH and file transfers reuse an authenticated connection for a while.

Connect:

```bash
ssh jureca
```

On first connection, compare the host fingerprint with JuDoor before accepting. The JURECA Ed25519 fingerprint shown during our setup was:

```text
SHA256:IDGkpGi9hxyNBixHDLEXYjLeGLrfNIiuTpcVXu0Lzcc
```

Enter your key passphrase and MFA code when requested. A successful login gives you a remote shell. `hostname` shows the remote machine; `exit` returns to your laptop.

On macOS, you can unlock your key in the local agent with:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_jureca
```

Use your actual key filename. [Official JURECA SSH instructions](https://apps.fz-juelich.de/jsc/hps/jureca/access.html)

## Join the shared environment once

**On JURECA, using your own account:** `jacquet2`, `brett2`, and `mondoloni1` each run:

```bash
source /p/scratch/training2666/delorme2/team51/teammate.sh
```

This script is already configured for our team. It sets your cache paths, checks access, and prepares your shell. The lead has already run the lead setup and does not run this teammate step.

Do not rerun `hackathon_setup.sh`, recreate the shared environments, or redownload the datasets. On future logins, activate the project and the task environment instead.

## Get the code and collaborate

**On your laptop:** accept the GitHub repository invitation, then clone:

```bash
git clone https://github.com/Florent765/hackaton.git
cd hackaton
git switch -c jacquet2/unlearning-baseline
```

Replace the example branch name with your username and work item. Suggested code folders are `unlearning/`, `model_equality/`, and `watermark/`; create them locally as needed. The remote watermark folder has the longer name shown below.

Before starting a new branch, update the team's default branch with `git pull --ff-only`. Commit focused changes on your branch:

```bash
git add unlearning/train.py
git commit -m "Add unlearning baseline"
git push -u origin HEAD
```

Open a pull request on GitHub for teammates to review and merge. Everyone uses a separate clone. Avoid having multiple people switch branches in one shared checkout.

Commit source code, small configuration files, and dependency specifications. Keep `.venv`, datasets, model weights, outputs, and API keys out of Git. Check `git status` before staging; prefer specific filenames over `git add .`.

## Choose a task

The shared layout is:

```text
/p/scratch/training2666/delorme2/team51/
├── unlearning/                            # Task files and .venv
├── model_equality/                        # Task files and .venv
├── Hackathon_Watermark_Localisation_Task/ # Task files and .venv
│   ├── models/                            # Downloaded starter models
│   └── release/                           # Extracted train and test data
├── teammate.sh
├── gpu_job.sh                             # Reusable Python GPU job launcher
├── gpu_check.sh                           # Environment validation job
└── output/                                # Validation logs
```

Read your task's README, starter code, and participant instructions before implementing a solution. Some models fetch additional weights on first use. Leaderboard API keys and submission rules come from the organizers; environment setup does not submit a solution.

**On JURECA, each new session:**

```bash
jutil env activate -p training2666
export TEAM_DIR=/p/scratch/training2666/delorme2/team51
cd "$TEAM_DIR/unlearning"
source .venv/bin/activate
```

Choose `model_equality` or `Hackathon_Watermark_Localisation_Task` instead when appropriate. The task environments differ, so use the matching `.venv`. [Project activation documentation](https://apps.fz-juelich.de/jsc/hps/jureca/environment.html)

## Upload code without overwriting teammates

Keep each person's uploaded code under `work/<username>/` inside the chosen task folder. Submit from the task root so the launcher finds its `.venv` and relative dataset paths.

**On your laptop, from the Git repo root:** this example uploads the contents of your local `unlearning/` folder. Change `JSC_USER` to your own Jülich username.

```bash
JSC_USER=jacquet2
ssh jureca "mkdir -p /p/scratch/training2666/delorme2/team51/unlearning/work/$JSC_USER"
rsync -av \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='output/' --exclude='.env' \
  unlearning/ \
  "jureca:/p/scratch/training2666/delorme2/team51/unlearning/work/$JSC_USER/"
```

The trailing slash on `unlearning/` copies its contents. This command does not delete remote files. Upload all code and configuration files your script needs, and use paths relative to the task root for shared data. Store credentials separately from uploaded code.

For watermark work, upload local `watermark/` to `Hackathon_Watermark_Localisation_Task/work/<username>/`. For model equality, use `model_equality/` on both sides.

Once a job is queued or running, leave the files it uses unchanged until it finishes. Slurm stores the batch script, but your Python files are read from disk at execution time. For concurrent experiments, upload each version into a separate run folder. [Slurm submission behavior](https://slurm.schedmd.com/sbatch.html)

## Submit a GPU job

**On JURECA:** the following assumes you uploaded a file named `train.py`. Replace that filename with your actual entry point.

```bash
jutil env activate -p training2666
export TEAM_DIR=/p/scratch/training2666/delorme2/team51
cd "$TEAM_DIR/unlearning"
mkdir -p output

sbatch ../gpu_job.sh "work/$USER/train.py"
```

Arguments are forwarded to Python if your program supports them:

```bash
sbatch ../gpu_job.sh "work/$USER/train.py" --epochs 10
```

The launcher requests one GPU, eight CPU cores, and a 30 minute limit through `training2666` and `cispahack`. It activates the task's `.venv` and runs `srun python -u` with your script and arguments. You do not need to activate the environment manually before submitting this launcher.

For a different runtime, override the limit for your job without editing the shared launcher:

```bash
sbatch --time=01:00:00 ../gpu_job.sh "work/$USER/train.py"
```

Use limits allowed by the organizers and reservation. A longer request can wait longer or be rejected. Requesting multiple GPUs only helps if your program supports using them.

`sbatch` returns a job ID. It queues the job; a successful submission does not mean your Python program has finished. You can disconnect after submitting. Run training and substantial computation through Slurm, keeping the login nodes available for editing and job management.

### Make Python use CUDA

Requesting a GPU allocates hardware. Your code still needs to use it. For PyTorch:

```python
import torch

assert torch.cuda.is_available(), "Run this through a GPU job"
device = torch.device("cuda")
model = model.to(device)
# Inside your training loop:
images = images.to(device)
labels = labels.to(device)
```

This assumes you have already created `model`, `images`, and `labels`. Keep task-specific framework configuration consistent with the starter code.

## Track jobs and save results

**On JURECA:** replace `15901234` with your submitted job ID.

```bash
squeue -u "$USER"
tail -f output/15901234.out
cat output/15901234.err
sacct -j 15901234 --format=JobID,State,ExitCode,Elapsed
scancel 15901234
```

Run `scancel` only when you want to stop that job. Press Ctrl+C to stop following a log; this does not cancel the job. Logs are under the task directory from which you submitted. They may not exist until the job starts.

| State | Meaning |
| --- | --- |
| `PD` | Waiting for resources or another condition |
| `CF` | Allocated nodes are being prepared |
| `R` | Running |
| `COMPLETED` | Finished successfully at the process level |
| `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY` | Inspect logs and adjust your program or resource request |

Completed jobs disappear from `squeue`; use `sacct` to inspect them afterward. Always check your metrics and saved files as well as the exit status.

Give every experiment a unique output folder. Your Python code can derive one from the job ID:

```python
import os
from pathlib import Path

run_dir = Path("output") / os.environ["USER"] / os.environ["SLURM_JOB_ID"]
run_dir.mkdir(parents=True, exist_ok=True)
# Save checkpoints, predictions, configuration, and metrics under run_dir.
```

Record the Git commit, command, hyperparameters, and result for useful runs. The Python example above is intended for a Slurm job, where those variables are provided.

**On your laptop:** download one run's results, replacing the username and job ID:

```bash
mkdir -p results
scp -r jureca:/p/scratch/training2666/delorme2/team51/unlearning/output/jacquet2/15901234 results/
```

Keep important code in GitHub and copy important results off scratch storage. Scratch is temporary storage; it is not our backup.

## Coordinate shared work

- Everyone submits jobs from their own account. The lead does not need to submit for the team.
- Share datasets and caches, but keep individual uploaded code and experiment outputs in separate folders.
- Agree with the team before installing or upgrading packages in a shared `.venv`. `uv run` can synchronize dependencies, so use the prepared launcher or activated Python environment for routine runs.
- Leave shared datasets, environments, permissions, and the common launcher intact. Ask `delorme2` to manage new team access.
- Upload only code from your laptop. Do not copy a laptop `.venv` onto JURECA.
- Check task submission instructions before sending leaderboard submissions. Save the exact configuration and code revision behind each submission.

## Troubleshoot common problems

| Problem | What to check |
| --- | --- |
| `Permission denied (publickey)` | Correct username and key; key uploaded under JURECA; activation delay; allowed source IP. If only IPv4 is allowed, try `ssh -4 jureca`. |
| MFA or `keyboard-interactive` failure | Connect in an interactive terminal and supply the current MFA code. |
| Changed Wi-Fi, VPN, or mobile network | Update the key's allowed source addresses in JuDoor. |
| Cannot access the team folder | Check project membership and ask `delorme2` to inspect permissions. |
| `jutil: command not found` on your laptop | SSH into JURECA first. |
| `.venv/bin/activate` missing | Submit from the task root, not your local repo or `work/<username>/`. |
| `ModuleNotFoundError` | Confirm the task environment; upload all required source files; coordinate missing dependencies with the team. |
| CUDA unavailable | Run through `sbatch` with the GPU launcher; check the error log. |
| Missing output logs | Create `output/` before submitting and check whether the job has started. |
| Job reaches its time limit | Save checkpoints periodically and request an appropriate runtime within the reservation limits. |
| Reservation or account rejected | Confirm `training2666`, `dc-gpu`, and `cispahack` with the organizers, especially after the reservation window. |

The initial validation job `15901130` completed successfully on 21 November 2026: both PyTorch task environments performed GPU tensor operations on an A100, and the model equality and watermark imports passed. To repeat those checks, submit from the shared team root:

```bash
cd /p/scratch/training2666/delorme2/team51
mkdir -p output
sbatch gpu_check.sh
```
