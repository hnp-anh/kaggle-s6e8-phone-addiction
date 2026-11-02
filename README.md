# Predict Phone Addiction (Kaggle Playground Series S6E8)

Binary classification of `addicted_label` from phone-usage features (screen time, social media, gaming, sleep,
notifications, app opens, …), scored by ROC-AUC. 691k training rows, 296k test rows, 12 features.

**Result:** logistic-regression stack of LightGBM + CatBoost. OOF AUC 0.96818, public LB 0.96947, **private LB 0.96910**
(the 1st place private score was 0.97176).

## Contents

| File | What it is |
|---|---|
| [`data_exploration.ipynb`](data_exploration.ipynb) | EDA: data overview, every feature vs the target, feature relationships, train/test drift, and a table of actionable conclusions |
| [`modeling.ipynb`](modeling.ipynb) | Missingness analysis, LightGBM feature ablations, class weighting, CatBoost variants, a diversity probe and the final ensemble |
| [`pipeline.py`](pipeline.py) | Shared training pipeline: features, nested out-of-fold target encoding and model definitions. Every model uses the same 5 folds and is cached to `artifacts/<model>.npz` |
| [`tune.py`](tune.py) | Optuna hyperparameter search for LightGBM / CatBoost |
| [`oscar/`](oscar/) | Slurm scripts for running the slow models on Brown's OSCAR cluster (see its README) |
| [`lessons_from_top_solutions.md`](lessons_from_top_solutions.md) | Post-competition comparison with the 1st and 2nd place solutions, and a reusable EDA process |

## Key findings

* **Exact values carry hidden signal.** `notifications_per_day`, `app_opens_per_day`, `sleep_hours` and `age` show no trend as
  numbers, but individual values have addiction rates from about 0.1 to 0.99 (an artefact of synthetic data generation).
  Out-of-fold target encoding of exact values was the biggest single gain (+0.003 AUC).
* **Screen time, weekend screen time and social media hours** are the strongest features (single-feature AUC 0.86–0.89),
  and the decision boundary is diagonal in screen time × social media, so ratio and difference features help tree models.
* Gender, stress level and academic impact are close to noise, and train and test are identically distributed, so local CV tracks the leaderboard.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e8 && unzip playground-series-s6e8.zip -d playground-series-s6e8

python pipeline.py --list    # available models and which are cached

# train in this order: lgb_final and cb_num reuse the best ablation's configuration (writes artifacts/<model>.npz)
python pipeline.py --models lgb_raw lgb_eng lgb_eng_te4 lgb_eng_te9 lgb_best+missing lgb_best+balanced \
    lgb_final cb_num cb_allcat logreg_probe
```

The CatBoost models are slow on a CPU; add `--device gpu` or use the `oscar/` scripts. Then run `modeling.ipynb`,
which loads the 10 cached models and builds the ensemble and `submission.csv`. The competition data, cached predictions (`artifacts/`)
and submission files are not included in this repository.
