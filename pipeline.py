"""Training pipeline for the phone-addiction models, shared by modeling.ipynb and batch jobs.

Every model uses the same 5 stratified folds. A trained model is cached as
artifacts/<name>.npz (OOF predictions, fold-averaged test predictions, per-fold AUC);
the notebook only reads these files, so models can be trained anywhere (e.g. on OSCAR)
and copied back.

    python pipeline.py --list
    python pipeline.py --models cb_num cb_allcat --device gpu
    python pipeline.py --models logreg_probe

Environment variables: DATA_DIR (competition CSVs), ART_DIR (cache), QUICK=1 (30k-row smoke test).
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", message=".*eval_set.*")

ROOT = Path(__file__).resolve().parent
QUICK = os.environ.get("QUICK") == "1"
DATA = Path(os.environ.get("DATA_DIR", ROOT / "playground-series-s6e8"))
ART = Path(os.environ.get("ART_DIR", ROOT / ("artifacts_quick" if QUICK else "artifacts")))
ART.mkdir(exist_ok=True)
TUNE_DIR = ROOT / ("tuning_quick" if QUICK else "tuning")
SEED, N_FOLDS, N_SEEDS = 42, 5, 3
# cores actually allocated (under Slurm, os.cpu_count() reports the whole node)
N_CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
TARGET = "addicted_label"

# ---------------------------------------------------------------- data
train = pd.read_csv(DATA / "train.csv", index_col=0)
test = pd.read_csv(DATA / "test.csv", index_col=0)
if QUICK:
    train, test = train.sample(30_000, random_state=0), test.sample(5_000, random_state=0)

y = train.pop(TARGET).to_numpy()
NUM = train.select_dtypes("number").columns.tolist()
CAT = [c for c in train.columns if c not in NUM]
RAW = NUM + CAT
FINGERPRINT = ["notifications_per_day", "app_opens_per_day", "sleep_hours", "age"]
prior = y.mean()
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))


# ---------------------------------------------------------------- features
def engineer(df):
    out = pd.DataFrame(index=df.index)
    screen = df["daily_screen_time_hours"].fillna(df["weekend_screen_time"] / 1.24)
    parts = df[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    out["screen_filled"] = screen
    out["parts_sum"] = parts
    out["other_hours"] = screen - parts
    out["social_share"] = df["social_media_hours"] / screen
    out["gaming_share"] = df["gaming_hours"] / screen
    out["work_share"] = df["work_study_hours"] / screen
    out["weekend_minus_daily"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    return out


def missing_features(df):
    out = df[RAW].isna().astype(np.int8).add_prefix("miss_")
    out["n_missing"] = out.sum(axis=1)
    return out


raw_all = pd.concat([train, test])
full = pd.concat([raw_all, engineer(raw_all), missing_features(raw_all)], axis=1)
for c in CAT:
    full[c] = full[c].astype("category")
ENG = engineer(raw_all.head()).columns.tolist()
MISS = missing_features(raw_all.head()).columns.tolist()
X_TR = full.iloc[:len(train)].reset_index(drop=True)
X_TE = full.iloc[len(train):].reset_index(drop=True)


def as_key(s):
    """Exact value as a string ("nan" for missing): target-encoding key and CatBoost categorical level."""
    return np.where(s.isna(), "nan", s.astype(object).astype(str))


KEY_TR = {c: as_key(train[c]) for c in RAW}
KEY_TE = {c: as_key(test[c]) for c in RAW}


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _te_map(keys, yv, smoothing, p):
    s = pd.DataFrame({"k": keys, "y": yv}).groupby("k")["y"].agg(["sum", "count"])
    return (s["sum"] + smoothing * p) / (s["count"] + smoothing)


TE_CACHE = None   # set to {} to memoise encodings (tuning re-encodes the same folds every trial)


def target_encode(cols, tr_idx, va_idx, smoothing=20, inner=5):
    """Nested OOF target encoding: inner-fold values for training rows, full-fold map for validation/test rows."""
    key = (tuple(cols), smoothing, inner, hash(tr_idx.tobytes()), hash(va_idx.tobytes()))
    if TE_CACHE is not None and key in TE_CACHE:
        return TE_CACHE[key]
    ytr = y[tr_idx]
    p = ytr.mean()
    inner_folds = list(StratifiedKFold(inner, shuffle=True, random_state=SEED + 1).split(tr_idx, ytr))
    enc_tr, enc_va, enc_te = {}, {}, {}
    for c in cols:
        k = KEY_TR[c][tr_idx]
        oof = np.empty(len(tr_idx))
        for a, b in inner_folds:
            oof[b] = pd.Series(k[b]).map(_te_map(k[a], ytr[a], smoothing, p)).fillna(p).to_numpy()
        full_map = _te_map(k, ytr, smoothing, p)
        enc_tr[f"te_{c}"] = oof
        enc_va[f"te_{c}"] = pd.Series(KEY_TR[c][va_idx]).map(full_map).fillna(p).to_numpy()
        enc_te[f"te_{c}"] = pd.Series(KEY_TE[c]).map(full_map).fillna(p).to_numpy()
    out = pd.DataFrame(enc_tr), pd.DataFrame(enc_va), pd.DataFrame(enc_te)
    if TE_CACHE is not None:
        TE_CACHE[key] = out
    return out


# ---------------------------------------------------------------- CV harness
RESULTS = {}


def cross_validate(name, fit_predict, force=False):
    """Run fit_predict on every fold; cache OOF + fold-averaged test predictions in ART/name.npz."""
    path = ART / f"{name}.npz"
    if path.exists() and not force:
        res = dict(np.load(path))
        print(f"{name}: loaded from cache, OOF AUC {roc_auc_score(y, res['oof']):.5f}", flush=True)
    else:
        oof, test_pred, fold_auc, iters = np.zeros(len(y)), np.zeros(len(X_TE)), [], []
        t0 = time.time()
        for k, (tr_idx, va_idx) in enumerate(folds):
            p_va, p_te, it = fit_predict(tr_idx, va_idx)
            oof[va_idx] = p_va
            test_pred += p_te / N_FOLDS
            fold_auc.append(roc_auc_score(y[va_idx], p_va))
            iters.append(it)
            print(f"  {name} fold {k}: AUC {fold_auc[-1]:.5f}  iters {it}  ({time.time() - t0:.0f}s)", flush=True)
        res = dict(oof=oof, test=test_pred, fold_auc=np.array(fold_auc), iters=np.array(iters),
                   secs=np.array(time.time() - t0))
        np.savez(path, **res)
        print(f"{name}: OOF AUC {roc_auc_score(y, oof):.5f}  ({res['secs']:.0f}s)", flush=True)
    RESULTS[name] = res
    return res


def load(name):
    """Cached result for a model that must already have been trained."""
    if name not in RESULTS:
        path = ART / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: train '{name}' first (python pipeline.py --models {name})")
        RESULTS[name] = dict(np.load(path))
    return RESULTS[name]


def paired_delta(a, b):
    """Mean and standard error of per-fold AUC differences (b − a); same folds, so the comparison is paired."""
    d = load(b)["fold_auc"] - load(a)["fold_auc"]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


# ---------------------------------------------------------------- models
LGB_PARAMS = dict(objective="binary", metric="auc", n_estimators=20 if QUICK else 20000, num_leaves=63,
                  min_child_samples=100, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_lambda=1.0, n_jobs=N_CPUS, verbose=-1, random_state=SEED)


def lgb_model(cols, te_cols=(), lr=0.1, neg_weight=1.0, params=None, smoothing=20, seeds=(SEED,)):
    """LightGBM fit_predict; with several seeds, predictions are averaged over one model per seed."""
    def fit_predict(tr_idx, va_idx, with_test=True):
        Xtr = X_TR.loc[tr_idx, cols].reset_index(drop=True)
        Xva = X_TR.loc[va_idx, cols].reset_index(drop=True)
        Xte = X_TE[cols]
        if te_cols:
            a, b, c = target_encode(te_cols, tr_idx, va_idx, smoothing=smoothing)
            Xtr, Xva, Xte = pd.concat([Xtr, a], axis=1), pd.concat([Xva, b], axis=1), pd.concat([Xte, c], axis=1)
        w = np.where(y[tr_idx] == 0, neg_weight, 1.0)
        p_va, p_te, iters = 0.0, 0.0, []
        for s in seeds:
            m = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {}), "learning_rate": lr, "random_state": s})
            m.fit(Xtr, y[tr_idx], sample_weight=w, eval_set=[(Xva, y[va_idx])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            p_va = p_va + m.predict_proba(Xva)[:, 1] / len(seeds)
            if with_test:
                p_te = p_te + m.predict_proba(Xte)[:, 1] / len(seeds)
            iters.append(m.best_iteration_)
        return p_va, p_te, int(np.mean(iters))
    return fit_predict


CB_PARAMS = dict(iterations=30 if QUICK else 20000, loss_function="Logloss", eval_metric="Logloss",
                 od_type="Iter", od_wait=200, use_best_model=True, l2_leaf_reg=3,
                 random_seed=SEED, verbose=0, allow_writing_files=False)
DEVICE = "cpu"   # set by --device; "gpu" trains CatBoost on CUDA


def catboost_model(frame_tr, frame_te, cat_cols, seeds=(SEED,), **params):
    """CatBoost fit_predict; with several seeds, predictions are averaged over one model per seed."""
    def fit_predict(tr_idx, va_idx, with_test=True):
        hw = dict(task_type="GPU", devices="0") if DEVICE == "gpu" else dict(thread_count=min(8, N_CPUS))
        tr_pool = Pool(frame_tr.loc[tr_idx], y[tr_idx], cat_features=cat_cols)
        va_pool = Pool(frame_tr.loc[va_idx], y[va_idx], cat_features=cat_cols)
        te_pool = Pool(frame_te, cat_features=cat_cols) if with_test else None
        p_va, p_te, iters = 0.0, 0.0, []
        for s in seeds:
            m = CatBoostClassifier(**{**CB_PARAMS, **hw, **params, "random_seed": s})
            m.fit(tr_pool, eval_set=va_pool)
            p_va = p_va + m.predict_proba(va_pool)[:, 1] / len(seeds)
            if with_test:
                p_te = p_te + m.predict_proba(te_pool)[:, 1] / len(seeds)
            iters.append(m.get_best_iteration())
        return p_va, p_te, int(np.mean(iters))
    return fit_predict


def keys_frame(keys, cols, prefix=""):
    return pd.DataFrame({f"{prefix}{c}": keys[c] for c in cols})


SPLINE_COLS = NUM + ["screen_filled", "other_hours", "social_share", "weekend_minus_daily"]


def logreg_model():
    """Different model family for the diversity check: splines + target encodings of all 12 columns."""
    def fit_predict(tr_idx, va_idx, with_test=True):
        a, b, c = target_encode(RAW, tr_idx, va_idx)
        def prep(X, enc):
            num = X[SPLINE_COLS].fillna(X[SPLINE_COLS].median())
            return pd.concat([num.reset_index(drop=True), logit(enc), X[MISS].reset_index(drop=True)], axis=1)
        Xtr, Xva, Xte = prep(X_TR.loc[tr_idx], a), prep(X_TR.loc[va_idx], b), prep(X_TE, c)
        ct = ColumnTransformer([("spl", SplineTransformer(n_knots=8, degree=3), SPLINE_COLS)], remainder=StandardScaler())
        m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=300))
        m.fit(Xtr, y[tr_idx])
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], 0
    return fit_predict


# ---------------------------------------------------------------- model registry
ABLATIONS = {
    "lgb_raw": dict(cols=RAW),
    "lgb_eng": dict(cols=RAW + ENG),
    "lgb_eng_te4": dict(cols=RAW + ENG, te_cols=FINGERPRINT),
    "lgb_eng_te9": dict(cols=RAW + ENG, te_cols=NUM),
}


def best_ablation():
    return max(ABLATIONS, key=lambda n: roc_auc_score(y, load(n)["oof"]))


def use_missing():
    """Include missing indicators only if they beat the best ablation by > 2 standard errors."""
    d, se = paired_delta(best_ablation(), "lgb_best+missing")
    return d > 2 * se


def final_te_cols():
    return ABLATIONS[best_ablation()].get("te_cols", ())


def final_lgb_config():
    """Columns and target-encoded columns of the final LightGBM (best ablation, + missing indicators if they help)."""
    best = ABLATIONS[best_ablation()]
    return dict(cols=best["cols"] + (MISS if use_missing() else []), te_cols=best.get("te_cols", ()))


def _final_lgb():
    return lgb_model(**final_lgb_config(), lr=0.03)


def cb_num_frames():
    """Train frame, test frame and categorical columns of cb_num."""
    fp_cols = list(final_te_cols()) or FINGERPRINT
    frame_tr = pd.concat([X_TR[NUM + ENG], keys_frame(KEY_TR, CAT), keys_frame(KEY_TR, fp_cols, "cat_")], axis=1)
    frame_te = pd.concat([X_TE[NUM + ENG], keys_frame(KEY_TE, CAT), keys_frame(KEY_TE, fp_cols, "cat_")], axis=1)
    return frame_tr, frame_te, CAT + [f"cat_{c}" for c in fp_cols]


# single-feature CTRs only: feature combinations are cb_allcat's job, and they make this model ~4x slower
CB_NUM_PARAMS = dict(learning_rate=0.06, depth=7, max_ctr_complexity=1)


def _cb_num():
    return catboost_model(*cb_num_frames(), **CB_NUM_PARAMS)


def tuned_config(model):
    """Best configuration found by tune.py (tuning/<model>_best.json)."""
    path = TUNE_DIR / f"{model}_best.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run tune.py --model {model} first")
    return json.loads(path.read_text())["config"]


def _lgb_tuned():
    cfg = tuned_config("lgb")
    return lgb_model(**final_lgb_config(), lr=0.02, params=cfg["params"], smoothing=cfg["smoothing"],
                     seeds=[SEED + i for i in range(N_SEEDS)])


def _cb_num_tuned():
    cfg = tuned_config("cb_num")
    return catboost_model(*cb_num_frames(), **{**CB_NUM_PARAMS, **cfg["params"], "learning_rate": 0.03},
                          seeds=[SEED + i for i in range(N_SEEDS)])


def _cb_allcat():
    return catboost_model(keys_frame(KEY_TR, RAW), keys_frame(KEY_TE, RAW), RAW,
                          learning_rate=0.08, depth=7, max_ctr_complexity=2)


MODELS = {
    **{name: (lambda cfg=cfg: lgb_model(**cfg)) for name, cfg in ABLATIONS.items()},
    "lgb_best+missing": lambda: lgb_model(cols=ABLATIONS[best_ablation()]["cols"] + MISS,
                                          te_cols=final_te_cols()),
    "lgb_best+balanced": lambda: lgb_model(**ABLATIONS[best_ablation()], neg_weight=prior / (1 - prior)),
    "lgb_final": _final_lgb,
    "cb_num": _cb_num,
    "cb_allcat": _cb_allcat,
    "logreg_probe": logreg_model,
    "lgb_tuned": _lgb_tuned,           # needs tuning/lgb_best.json
    "cb_num_tuned": _cb_num_tuned,     # needs tuning/cb_num_best.json
}


def train_model(name, force=False):
    return cross_validate(name, MODELS[name](), force=force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[], help="models to train, in order (see --list)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="CatBoost device")
    ap.add_argument("--force", action="store_true", help="retrain even if a cached result exists")
    ap.add_argument("--list", action="store_true", help="list models and whether they are cached")
    args = ap.parse_args()
    DEVICE = args.device
    print(f"train {train.shape}, test {test.shape}, cache {ART}, CatBoost device {DEVICE}, cpus {N_CPUS}", flush=True)
    if args.list or not args.models:
        for name in MODELS:
            print(f"  {name:20s} {'cached' if (ART / f'{name}.npz').exists() else '-'}")
    for name in args.models:
        train_model(name, force=args.force)
