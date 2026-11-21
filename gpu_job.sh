#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# Submit from the task directory, after creating output/:
#   mkdir -p output
#   sbatch ../gpu_job.sh your_script.py [arguments...]
set -euo pipefail
: "${1:?Usage: sbatch ../gpu_job.sh your_script.py [arguments...]}"
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export HF_HOME=/p/scratch/training2666/delorme2/team51/.cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
srun python -u "$@"
