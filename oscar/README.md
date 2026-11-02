# Training the remaining models on OSCAR

The LightGBM models are already trained and cached in `artifacts/`. This job trains the slow ones:
`cb_num` and `cb_allcat` (CatBoost on a GPU) and `logreg_probe` (CPU). Each writes
`artifacts/<model>.npz`, which `modeling.ipynb` then loads, so nothing is retrained locally.

Replace `USER` with your OSCAR username. Run the laptop commands from the project folder:
`cd ~/Documents/"Kaggle Predict Phone Addiction "` (the folder name ends with a space).

## 1. Upload (laptop, about 120 MB)
```bash
rsync -av --exclude .venv --exclude __pycache__ --exclude '*.ipynb' --exclude .DS_Store \
    ./ USER@ssh.ccv.brown.edu:~/phone-addiction/
```
This uploads the data, `pipeline.py`, `oscar/` and the cached LightGBM results. `cb_num` needs those results:
it copies the target-encoded columns of the best LightGBM.

## 2. Set up and submit (on OSCAR)
```bash
ssh USER@ssh.ccv.brown.edu
cd ~/phone-addiction
bash oscar/setup_env.sh                 # once: venv in ~/venvs/phone-addiction (about 3 min)

# optional 2-minute GPU smoke test on 30k rows (writes to artifacts_quick/, not artifacts/)
sbatch --export=ALL,QUICK=1,MODELS=cb_allcat oscar/train_gpu.slurm

sbatch oscar/train_gpu.slurm            # the real run: cb_num, cb_allcat, logreg_probe
squeue -u $USER                         # job status
tail -f logs/phone-addiction-<jobid>.out
```
If `setup_env.sh` can't find Python 3.11+, run `module avail python` and re-run it with
`PY_MODULE=python/<version> bash oscar/setup_env.sh`.

To run a subset, or to re-run after a failure (finished models are cached and skipped):
`sbatch --export=ALL,MODELS="cb_allcat logreg_probe" oscar/train_gpu.slurm`.
Add `--force` to the python line in `train_gpu.slurm` to retrain cached models.

## 3. Download the results (laptop)
```bash
rsync -av USER@ssh.ccv.brown.edu:~/phone-addiction/artifacts/ ./artifacts/
```
Then run `modeling.ipynb`. It loads all 10 models from `artifacts/`, then builds the ensemble and `submission.csv`.

## Notes
* GPU CatBoost uses a different implementation from CPU CatBoost, so its scores can differ slightly from a laptop run.
  This doesn't matter for the ensemble, because all models still share the same folds.
* `python pipeline.py --list` shows which models are cached.

# Tuning (`tune.py`)

This tunes `lgb_final` (CPU node) and `cb_num` (GPU node) with Optuna, then trains each one's best configuration
at a low learning rate, averaged over 3 seeds. The results are `artifacts/lgb_tuned.npz` and `artifacts/cb_num_tuned.npz`.

```bash
# laptop: upload the new code (same command as before)
rsync -av --exclude .venv --exclude __pycache__ --exclude '*.ipynb' --exclude .DS_Store \
    --exclude artifacts_quick --exclude tuning_quick \
    ./ USER@ssh.ccv.brown.edu:~/phone-addiction/

# OSCAR: add optuna to the existing environment (once)
source ~/venvs/phone-addiction/bin/activate && pip install optuna==5.0.0 && deactivate

# OSCAR: submit both jobs; they run in parallel (default 5 h of search, then the final model; 8 h limit)
cd ~/phone-addiction
sbatch oscar/tune_lgb.slurm
sbatch oscar/tune_cb.slurm
tail -f logs/tune-lgb-<jobid>.out      # one line per trial: its score (or "pruned"), the best so far, the baseline

# laptop: download the results
rsync -av USER@ssh.ccv.brown.edu:~/phone-addiction/artifacts/ ./artifacts/
rsync -av USER@ssh.ccv.brown.edu:~/phone-addiction/tuning/ ./tuning/
```

* **Resuming:** the search is saved in `tuning/<model>.db`. Submitting the same job again continues the same search rather than starting over.
* **Out of time during the final model:** `sbatch --export=ALL,TUNE_HOURS=0 oscar/tune_lgb.slurm` skips the search and
  only trains the final model from `tuning/lgb_best.json`. The same works for `tune_cb.slurm`.
* **Progress at any time:** `cat tuning/lgb_best.json` shows the best score and settings so far, next to the baseline score.
