#!/bin/bash
# Trains the remaining models on one GPU node. Submit from the project folder:
#   sbatch oscar/train_gpu.slurm
# Override the model list:  sbatch --export=ALL,MODELS="cb_allcat" oscar/train_gpu.slurm
#SBATCH --job-name=phone-addiction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODELS="${MODELS:-cb_num cb_allcat logreg_probe}"
VENV=~/venvs/phone-addiction

module load "$(cat "$VENV/.python_module")"
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), models: $MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u pipeline.py --models $MODELS --device gpu
python -u pipeline.py --list
