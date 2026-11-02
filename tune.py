"""Optuna hyperparameter search for the two strong models, on the same 5 folds as pipeline.py.

Each trial trains at a fast learning rate and reports its running mean fold AUC after every fold,
so clearly worse trials are pruned early. The first trial is the current configuration, so the
search can only match or beat it. The study is stored in tuning/<model>.db, so a re-submitted job
resumes where the previous one stopped; the best configuration is written to tuning/<model>_best.json
after every trial, which pipeline.py reads for lgb_tuned / cb_num_tuned.

    python tune.py --model lgb --hours 4
    python tune.py --model cb_num --device gpu --hours 4
"""
import argparse
import json
import time

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

import pipeline as pl

TUNE_LR = {"lgb": 0.1, "cb_num": 0.1}


def lgb_trial(trial):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        max_bin=trial.suggest_categorical("max_bin", [127, 255, 511]),
    )
    smoothing = trial.suggest_categorical("te_smoothing", [5, 20, 100])
    fit_predict = pl.lgb_model(**pl.final_lgb_config(), lr=TUNE_LR["lgb"], params=params, smoothing=smoothing)
    return fit_predict, {"params": params, "smoothing": smoothing}


LGB_BASELINE = dict(num_leaves=63, min_child_samples=100, colsample_bytree=0.7, subsample=0.8,
                    reg_lambda=1.0, reg_alpha=1e-3, max_bin=255, te_smoothing=20)

CB_FRAMES = None


def cb_num_trial(trial):
    global CB_FRAMES
    if CB_FRAMES is None:
        CB_FRAMES = pl.cb_num_frames()
    params = dict(
        depth=trial.suggest_int("depth", 5, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        border_count=trial.suggest_categorical("border_count", [128, 254]),
    )
    fit_predict = pl.catboost_model(*CB_FRAMES, **{**pl.CB_NUM_PARAMS, **params, "learning_rate": TUNE_LR["cb_num"]})
    return fit_predict, {"params": params}


CB_BASELINE = dict(depth=7, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0, border_count=128)

SPACES = {"lgb": (lgb_trial, LGB_BASELINE), "cb_num": (cb_num_trial, CB_BASELINE)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(SPACES), required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="CatBoost device")
    ap.add_argument("--hours", type=float, default=4.0, help="stop starting new trials after this long")
    ap.add_argument("--trials", type=int, default=200, help="maximum number of trials in this run")
    args = ap.parse_args()

    pl.DEVICE = args.device
    pl.TE_CACHE = {}
    pl.TUNE_DIR.mkdir(exist_ok=True)
    make_trial, baseline = SPACES[args.model]

    study = optuna.create_study(
        study_name=args.model, direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{pl.TUNE_DIR / f'{args.model}.db'}",
        sampler=optuna.samplers.TPESampler(seed=pl.SEED, multivariate=True, n_startup_trials=12),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    if not study.trials:
        study.enqueue_trial(baseline, user_attrs={"baseline": True})

    def objective(trial):
        fit_predict, config = make_trial(trial)
        trial.set_user_attr("config", config)
        t0, aucs = time.time(), []
        for k, (tr_idx, va_idx) in enumerate(pl.folds):
            p_va, _, iters = fit_predict(tr_idx, va_idx, with_test=False)
            aucs.append(roc_auc_score(pl.y[va_idx], p_va))
            trial.report(float(np.mean(aucs)), k)   # running mean over the same folds: comparable across trials
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("secs", time.time() - t0)
        return float(np.mean(aucs))

    def save_best(study, trial):
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            return
        best = study.best_trial
        baseline_value = next((t.value for t in done if t.user_attrs.get("baseline")), None)
        (pl.TUNE_DIR / f"{args.model}_best.json").write_text(json.dumps({
            "config": best.user_attrs["config"], "mean_fold_auc": best.value, "trial": best.number,
            "baseline_mean_fold_auc": baseline_value, "tune_learning_rate": TUNE_LR[args.model],
            "n_complete": len(done), "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        }, indent=2))
        study.trials_dataframe().to_csv(pl.TUNE_DIR / f"{args.model}_trials.csv", index=False)
        state = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"{trial.value:.5f}"
        print(f"trial {trial.number}: {state}  |  best {best.value:.5f} (trial {best.number}), "
              f"baseline {baseline_value if baseline_value is None else round(baseline_value, 5)}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"tuning {args.model} on {pl.N_CPUS} cpus, device {args.device}, up to {args.trials} trials / {args.hours} h", flush=True)
    study.optimize(objective, n_trials=args.trials, timeout=args.hours * 3600, callbacks=[save_best])
    print(json.dumps(json.loads((pl.TUNE_DIR / f"{args.model}_best.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
