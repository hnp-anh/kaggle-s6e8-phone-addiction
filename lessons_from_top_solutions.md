# Lessons from S6E8: top solutions and a reusable EDA process

Sources: the 1st place writeup (Chris Deotte, NVIDIA) and the 2nd place writeup (Xin Feng), compared with our pipeline
(`pipeline.py`, `modeling.ipynb`). Written 2026-10-31, after the competition ended.
Section 7 generalises the EDA in `data_exploration.ipynb` into a process for future tabular tasks.

## 1. Where we stand

| | OOF AUC | Public | Private |
|---|---|---|---|
| **Ours**: LR stack of `lgb_final` + `cb_num` | 0.96818 | 0.96947 | 0.96910 |
| 2nd place, week 1: 100–200 basic models, blended | 0.97013 | 0.97120 | 0.97096 |
| 2nd place, final blend | 0.97055 | 0.97148 | 0.97123 |
| 1st place, **single RealMLP** | 0.97070 | 0.97174 | not reported (1st place private: 0.97176) |
| 1st place, single XGBoost | 0.97020 | 0.97030 | – |

**OOF is a reliable guide.** For both us and the 2nd place team, private LB ≈ OOF + 0.0007 to 0.0009. So reaching the
top-100 band (private ≈ 0.9711) needs about **0.9703 OOF**, which is **+0.002 over ours**.

**The gap is in the single models, not the ensemble.** The 1st place solution won with *one* model (RealMLP, OOF 0.97070),
which beats the 2nd place blend of hundreds of models. Our best single model is 0.96797; theirs are 0.9702–0.9707.
Tuning and seeds (the current OSCAR jobs) are worth about +0.0003, so the rest must come from **features** and from **model types we don't have yet**.

## 2. What the top solutions did that we don't (yet)

| Idea | Source | Do we have it? |
|---|---|---|
| Exact-value target encoding of raw columns ("RAW_TE") in XGB/LGB | 2nd | **Yes**: nested OOF TE of all 9 numerics (+0.0028 AUC, our biggest gain) |
| Raw values as categorical ("object") columns in CatBoost/RealMLP | 2nd | **Partly**: `cb_num` has categorical copies of the 9 numerics; no NN |
| Ratio features | 2nd | **Partly**: social/gaming/work shares of screen time |
| `fake_daily = social + work + game` | 2nd | **Yes**: `parts_sum`. But we don't treat it as a RAW feature (no TE, bins or ratios of it) |
| `fake_social = daily − work − game`, `fake_work = daily − social − game`, `fake_game = daily − social − work` | 2nd | **No** (we only have `other_hours = screen − parts`) |
| Treat the fake features as RAW: TE, bins, ratios | 2nd | **No** |
| **Digit** features | 2nd | **No** |
| **Round** features | 2nd | **No** |
| **Bins** | 2nd | **No** |
| **Original dataset**, both as extra columns ("as COL") and as extra rows ("as ROW") | 2nd | **No** |
| XGBoost | 1st, 2nd | **No** (best single XGB: OOF 0.97020) |
| **RealMLP** (tabular neural net) | 1st, 2nd | **No**: it was the *best single model* in 1st place |
| Big zoo of models, each = RAW + one feature group | 1st (380 models), 2nd (200–400) | No: we have 4 models |
| Logistic regression as the blender | 2nd | **Yes** |
| Seeds / Optuna tuning | 2nd | **In progress** on OSCAR |
| Public notebook predictions as extra blend members | 2nd (200+ public predictions) | No. A gamble; public LB overfitting risk |

The 2nd place team: *"It became relatively easy to get around 0.969 OOF and 0.970+ LB with XGBoost, LightGBM,
CatBoost, and RealMLP"* once the fake features were treated as RAW features. That is +0.001 over our best single model,
from one cheap idea.

## 3. Corrections to my earlier conclusions

1. **"A neural net is unlikely to add anything" was wrong.** I based it on our logistic-regression probe. In 1st place,
   **RealMLP was the strongest single model** (0.97070 OOF), ahead of XGBoost (0.97020). A well-built tabular NN is not
   like logistic regression here. RealMLP (`pytabkit` package) belongs in the plan, trained on a GPU on OSCAR.
2. **"The original dataset probably isn't the answer" was too strong.** The repo I cited said adding it as rows hurt.
   But the 2nd place team used it in many basic models, both as columns and as rows. It is a feature *source*, not just extra rows.
3. **"Spend effort on tuning and seeds" was the wrong priority.** Both writeups say the opposite. The 2nd place team: *"I thought
   I needed a better blender. In fact, I needed a better model."* Tune *after* the feature set stops changing,
   because tuned parameters may not carry over to new features.

## 4. Things we already do right (keep them)

- Trust OOF, use public LB only as a sanity check. Never tune on public LB (2nd place, section 4.4).
- Nested, leak-free target encoding inside each fold.
- Logistic-regression stacking on OOF predictions.
- One fixed set of folds for every model, so OOF predictions can be stacked.
- Baselines in line with the 2nd place team's guide (LR ≈ 0.960, XGB/LGB ≈ 0.966 without feature engineering): ours are 0.961 and 0.9635 raw / 0.9674 with TE.

## 5. Plan, in order

Each step is checked with the fast LightGBM ablation (learning rate 0.1, about 4 min locally, identical folds, paired
per-fold deltas). A step is kept only if its gain is > 2 standard errors.

1. **Fake features as RAW features.** Add `fake_social`, `fake_work`, `fake_game` (we already have `fake_daily` as `parts_sum`).
   Then give *all four* the same treatment as the raw columns: exact-value TE, ratios to screen time, and quantile bins.
   Cheap, and this is what took the 2nd place single models to about 0.969.
2. **Round and digit features.** For each numeric column: the value rounded to 1 and 0 decimals, plus TE of the rounded
   values; and digit features, e.g. `int(x * 100) % 10`, `int(x * 10) % 10`, `int(x) % 10`. The writeup doesn't define them;
   these are the usual Playground versions. Synthetic generators leave patterns in the low digits.
3. **Original dataset.** First find out whether we can still get it (it has been removed from Kaggle, so it may survive
   only as a mirror or in public notebooks). "As COL": for each exact value of each column, the original data's mean target
   (and count), added as features. "As ROW": the original rows appended to training, flagged with an `is_original` column.
   Test both separately.
4. **XGBoost** on the best feature set. The 1st place single XGB reached 0.97020; the 2nd place team regrets not pushing one strong XGB further.
5. **RealMLP** on GPU (OSCAR): numerics plus raw values as categorical columns. This was the 1st place winning single model.
6. **Then** Optuna tuning and seed averaging on the final feature set, and an LR stack over LGB / XGB / CatBoost / RealMLP.
7. Optional: a small model zoo of "RAW + one feature group" variants for stack diversity. Only after steps 1–5, since the 2nd place team found blending had the least return.

**Status of the current OSCAR tuning jobs:** they tune on the *old* feature set. Let them finish (it's a cheap +0.0002 to +0.0003 now),
but re-tune once steps 1–3 change the features.

## 6. Process lessons

- **One idea per experiment, one notebook or script per purpose:** feature engineering, Optuna, seeds/folds, and new
  ideas each separate (2nd place, section 4.3). Our `pipeline.py` registry plus cached artifacts already supports this.
- **Improve the strongest single model before building diversity.** Blending hundreds of similar models plateaued for the 2nd place team ("Week 4: nothing works").
- **In AUC competitions, ranks swing by ±100 places on noise of about 0.0001.** Pick the final submission by OOF, not by public LB.
- **AI agents as collaborators were the 1st place's edge.** Several agents competed on the best single NN and the best
  single XGB, sharing tips when one fell behind; a third model was asked for new feature ideas. The specific new
  features from that work were not disclosed.

## 7. A reusable EDA process for tabular prediction tasks

This is the process behind `data_exploration.ipynb`, written so it can be reused. It also includes the fixes for what
that EDA got wrong or missed (marked **Fix**). It is written for binary classification but carries over to regression
(replace "target rate" with "target mean") and to multiclass (one rate per class).

### 7.1 Principles

1. **Every chart answers a question, and every section ends in a Takeaway with an Action.** A Takeaway names the
   finding, the evidence (a number, not "looks like"), and what to do in modeling, even if the action is "nothing".
   A chart with no decision attached is decoration.
2. **Pair every chart with a statistic.** Single-feature AUC, a rate with its confidence interval, a test with its
   p-value, a correlation. The chart shows the shape; the number decides.
3. **Significant is not the same as useful.** With hundreds of thousands of rows, tiny effects pass tests. Ask two
   separate questions: is it real (test), and does it move CV (ablation)? Only the second decides what goes in the model.
4. **Close the loop with a quick model.** End the EDA with a cheap model on a sample: raw features vs raw + each
   conclusion, on the same folds, compared with paired per-fold deltas. Conclusions that don't move CV get demoted.
5. **Finish with one table: finding → evidence (section) → action.** This becomes the modeling plan.
6. **Keep visual conventions fixed:** one colour per class used everywhere (here blue = 0, orange = 1),
   dot plots instead of bars on a truncated axis, gridlines behind the data, the same feature order in every grid of plots.
7. **Build the notebook from a script** (cells in a `.py` file, executed headless) so it can be regenerated after a fix.

### 7.2 The steps, in order

| # | Question | Charts | Statistic | Typical action |
|---|---|---|---|---|
| 0 | What exactly is predicted and scored? | none | rows, columns, dtypes, number of unique values per column, metric, submission format | Choose the metric to optimise and the CV scheme. A low unique-value count on a "numeric" column is the first hint for step 4. |
| 1 | How balanced is the target? | bar of class counts | class shares | StratifiedKFold. **Don't decide reweighting here:** list it as an ablation (here it hurt AUC and broke calibration). |
| 2 | How much is missing, where, and is it informative? | missing rate per column (train vs test); co-occurrence heatmap of missingness; target rate when missing vs present, with CIs; target rate by number of gaps per row | see 7.3 | Native NaN for GBMs. Indicators only if the ablation says so. Impute from correlated columns for linear models and NNs. |
| 3 | Which features separate the classes on their own? | per-class distribution (KDE or histogram) for every numeric | single-feature AUC (folded: max(AUC, 1 − AUC)) | Rank the features. Note ranges and outliers: transforms only if the data needs them. |
| 4 | **Does the exact value matter beyond its trend?** | target rate in ~20 quantile bins (line) with the rate of each exact value (dots, min count ~200) on top | spread of per-value rates around the bin line; out-of-fold target-encoded AUC vs raw AUC | If the dots scatter off the line: treat the column as categorical too (OOF target encoding, count encoding, `category` dtype). **This was the biggest single finding here (+0.003 AUC).** |
| 5 | Which features matter, summarised? | dot plot: raw AUC vs OOF-TE AUC per feature | the two AUCs | Feature tiers; candidates to drop. The TE must be out-of-fold, or this plot lies. |
| 6 | Do the categoricals matter? | target rate per level with CIs | rate difference, AUC | Native or one-hot encoding; near-flat levels are drop candidates (test in CV). |
| 7 | How do features relate to each other? | correlation heatmap; hexbin of strong pairs coloured by target rate | r, and the median ratio or difference of strong pairs | Impute from partners; ratio, sum and difference features; diagonal boundaries mean ratio features help trees. **Fix:** see 7.4. |
| 8 | **How are the numbers written?** (new) | number of decimals per column; target rate per last digit and per rounding residual | rate spread across digits | Digit and rounding features. Synthetic generators leave patterns there. This EDA skipped this step; 2nd place used it. |
| 9 | Do train and test match? | per-feature overlays; missing rates side by side | mean differences; share of test values unseen in train; adversarial-validation AUC if anything looks off | Trust CV if they match. Unseen test values limit what target encoding can do. |
| 10 | **Is there source or external data?** (new) | per-value target rates in the original data vs in train | correlation of the two rate tables | Original data as columns (per-value original target rate and count) and/or as rows with a flag. This EDA skipped this step; 2nd place used it. |
| 11 | Do the conclusions improve a model? | dot plot of CV AUC per variant | paired per-fold deltas ± SE | Keep what beats 2 SE; drop the rest from the plan. |
| 12 | What's the plan? | conclusions table | none | Feeds the modeling notebook directly. |

### 7.3 Missingness: the four checks (Fix)

The original EDA concluded "missing completely at random" because the target rate barely moved when a value was missing.
That inference is wrong: **the same target rate with and without a value does not make it random.** You noticed the small,
consistent shift; the proper checks showed the gaps are structured. Run all four:

1. **Co-occurrence.** Correlate the missing indicators. Here gaps came in blocks ({daily, weekend, social, gaming},
   {sleep, stress, academic}, {notifications, app opens}, {age, gender}), so they are not independent across columns.
2. **Dependence on observed values (MAR).** Missing rate as a function of other features. Here edge ages (18–21, 34–35) had more gaps.
3. **Dependence on the target.** Target rate missing vs present per column with CIs, plus one pooled test
   (here Spearman(n_missing, y) = 0.0028, p = 0.019: real but tiny).
4. **Does it help a model?** Add indicators and `n_missing` to the best GBM and compare per fold
   (here +0.00002 ± 0.00003: no). This check decides; checks 1–3 explain.

State the conclusion in those terms: "structured (MAR), weakly related to the target, not useful as a feature for GBMs".

### 7.4 Relationships between features: go one step further (Fix)

The EDA found that screen time ≈ social + gaming + work + a gap (r = 0.84) and made share and difference features.
It stopped there. The step that worked for 2nd place:

* **For each near-identity, rebuild every term from the others** (`fake_social = daily − work − game`, and so on),
  not just the one residual.
* **Then treat each derived feature like a raw column:** run step 4 on it (per-value rate vs trend), target-encode it,
  bin it, and take ratios. Derived features in synthetic data can carry exact-value signal just like raw ones.

General rule: when a relationship between columns is strong, the derived columns deserve the same full EDA pass as the raw ones.

### 7.5 Synthetic (Kaggle Playground) data: extra checks

Playground data is generated from a small original dataset, which leaves artefacts that real data doesn't have:

* exact values copied with their labels (step 4: per-value rates far off the trend);
* digit and rounding patterns (step 8);
* an original dataset whose per-value statistics transfer (step 10);
* train and test drawn from the same generator, so CV tracks the leaderboard closely (here private ≈ OOF + 0.0009).

Run steps 4, 8 and 10 first on Playground data: that is where the signal hides.

### 7.6 Copyable checklist

```
[ ] 0  task, metric, dtypes, unique counts
[ ] 1  target balance -> CV scheme (reweighting = ablation, not a decision)
[ ] 2  missingness: rates, co-occurrence, MAR, target link, ablation
[ ] 3  per-class distributions + single-feature AUC
[ ] 4  binned trend vs exact-value rates  (every numeric, incl. derived)
[ ] 5  raw AUC vs OOF-TE AUC summary -> tiers
[ ] 6  categorical level rates with CIs
[ ] 7  correlations, strong pairs, identities -> rebuild every term, treat as raw
[ ] 8  decimals, digits, rounding vs target
[ ] 9  train/test drift, unseen values
[ ] 10 original/external data: per-value rates, as COL and as ROW
[ ] 11 quick model: raw vs +each conclusion, paired fold deltas
[ ] 12 finding -> evidence -> action table
```
