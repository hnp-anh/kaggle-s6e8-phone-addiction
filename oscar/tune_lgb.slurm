#!/bin/bash
# Tunes LightGBM on a CPU node, then trains lgb_tuned (3 seeds, learning rate 0.02).
#   sbatch oscar/tune_lgb.slurm
# Final model only (search already done): sbatch --export=ALL,TUNE_HOURS=0 oscar/tune_lgb.slurm
# Shorter/longer search:  sbatch --export=ALL,TUNE_HOURS=2 oscar/tune_lgb.slurm
#SBATCH --job-name=tune-lgb
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TUNE_HOURS="${TUNE_HOURS:-5}"
VENV=~/venvs/phone-addiction

module load "$(cat "$VENV/.python_module")"
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, tuning for $TUNE_HOURS h"
if [ "$TUNE_HOURS" != "0" ]; then
    python -u tune.py --model lgb --hours "$TUNE_HOURS"
fi
python -u pipeline.py --models lgb_tuned
