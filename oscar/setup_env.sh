#!/bin/bash
# One-time setup on OSCAR (run on a login node from the project folder):  bash oscar/setup_env.sh
# Creates ~/venvs/phone-addiction with the same package versions as the local .venv.
set -euo pipefail

# pandas 3 needs Python >= 3.11. Override with: PY_MODULE=python/<version> bash oscar/setup_env.sh
PY_MODULE="${PY_MODULE:-}"
if [ -z "$PY_MODULE" ]; then
    PY_MODULE=$(module -t avail python 2>&1 | grep -E '^python/3\.(1[1-9])' | sort -V | tail -1 || true)
fi
if [ -z "$PY_MODULE" ]; then
    echo "No python/3.11+ module found. Run 'module avail python' and re-run with PY_MODULE=python/<version>." >&2
    exit 1
fi
echo "Using module $PY_MODULE"
module load "$PY_MODULE"
python3 --version

VENV=~/venvs/phone-addiction
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"
echo "$PY_MODULE" > "$VENV/.python_module"
python -c "import catboost, lightgbm, pandas, sklearn; print('catboost', catboost.__version__, '| lightgbm', lightgbm.__version__, '| pandas', pandas.__version__)"
mkdir -p "$(dirname "$0")/../logs"   # Slurm writes job logs here; it must exist before sbatch
echo "Environment ready: $VENV"
