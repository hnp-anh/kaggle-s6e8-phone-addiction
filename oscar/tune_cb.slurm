#!/bin/bash
# Tunes cb_num on a GPU node, then trains cb_num_tuned (3 seeds, learning rate 0.03).
#   sbatch oscar/tune_cb.slurm
# Final model only (search already done): sbatch --export=ALL,TUNE_HOURS=0 oscar/tune_cb.slurm
# Shorter/longer search:  sbatch --export=ALL,TUNE_HOURS=2 oscar/tune_cb.slurm
#SBATCH --job-name=tune-cb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TUNE_HOURS="${TUNE_HOURS:-5}"
VENV=~/venvs/phone-addiction

module load "$(cat "$VENV/.python_module")"
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), tuning for $TUNE_HOURS h"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$TUNE_HOURS" != "0" ]; then
    python -u tune.py --model cb_num --device gpu --hours "$TUNE_HOURS"
fi
python -u pipeline.py --models cb_num_tuned --device gpu
