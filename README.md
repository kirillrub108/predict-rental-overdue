# Midterm Project — forecasting overdue bike rental returns

The goal is to predict, at the moment a bike is handed out, whether it will come back late:
more than 15 minutes after the planned return time (`y_overdue`).

**Status:** exploratory data analysis, baselines, two tuned models and the final test evaluation
are done (`main.ipynb`).

| Path | What it is |
|---|---|
| `main.ipynb` | notebook: data loading, EDA, conclusions, baselines, models, tuning, best model |
| `dataset.csv` | one row is one checkout (39,344 rows, 27 columns, 2025-01-18 to 2026-09-30): features known at checkout time and the targets `y_overdue`, `y_lost` |
| `requirements.txt` | Python dependencies |

The data is synthetic: the signal is baked in by the generator, so the metrics of any model
trained on it will not carry over to real data. The generator is not part of this repository.

## Columns

| Group | Columns |
|---|---|
| Keys | `id`, `started_at`, `customer_id`, `client_id` |
| Numeric | `hour`, `month`, `plan_hours`, `n_batteries`, `deposit_amount`, `price_total`, `prepaid_share`, `client_prev_rentals`, `client_prev_late_rate`, `client_prev_lost`, `days_since_prev` |
| Categorical | `tenant`, `dow`, `model`, `deposit_kind`, `client_residency`, `issued_by_employee_id` |
| Boolean | `is_weekend`, `block_missing`, `first_visit`, `prev_late` |
| Targets | `y_overdue`, `y_lost` |

## Target

Only overdue returns are modelled. Lost bikes (`y_lost`, 391 checkouts, about 1%) have no `y_overdue`
label, so they are dropped together with the `y_lost` column. The remaining 38,953 checkouts have an
overdue rate of 8.3%, so the classes are strongly imbalanced.

## EDA (in `main.ipynb`)

1. Load the dataset
2. Base checks (shape, duplicates, dtypes) and nulls; drop lost bikes
3. Describe data: numeric (quantiles, skew, kurtosis, outliers), categorical, boolean
4. Visualize: class balance and the target over time
5. Correlation with the target: single-feature ROC AUC, overdue rate by group (lift and z-score),
   dependence on continuous features, effect of missing values
6. Correlation matrices: Spearman for numeric and boolean features, Cramér's V for categorical ones,
   leakage check

Main findings (see the General Conclusion at the end of the notebook):

- Use PR AUC as the main metric. Split by time instead of a stratified split: the overdue rate drifts
  over time and client-history features are built from the past, so shuffling would leak the future
  into training.
- The strongest signals are `plan_hours` and `price_total`, then `deposit_kind`, `client_residency`,
  `prepaid_share`, `first_visit`, weekends and the evening hours.
- Missing client history is informative (10.5% overdue vs 6.8%), gaps in the card block look random.
- No leakage found; `client_prev_lost`, `issued_by_employee_id` and one of `first_visit` /
  `client_prev_rentals` should be dropped as constant or redundant.

## Modelling (in `main.ipynb`)

1. Final cleaning: drop `first_visit` and `issued_by_employee_id` (and the constant `client_prev_lost`).
2. Time-based split without shuffling: first 60% of checkouts for train, next 20% for validation, last 20% for test.
   The test set is used only for the final evaluation.
3. Baselines: the constant base rate and a logistic regression on `plan_hours` alone.
4. Logistic regression with L2: median imputation with missing-value indicators and scaling for numeric
   features, one-hot encoding for categorical ones; `C` is chosen by PR AUC on the validation set.
5. `HistGradientBoostingClassifier`: categories as integer codes, missing values handled by the model;
   hyperparameters are tuned in two rounds of random search over 4 time-based folds (`TimeSeriesSplit`)
   on train + validation, and the number of trees is chosen on the same folds.
6. Best model: `make_transformer`, `make_model`, `train` and `predict` in the last cells of the notebook;
   the model is refit on train + validation and evaluated once on the test set.

Test results (overdue rate in the test set: 7.6%):

| Model | PR AUC | ROC AUC |
|---|---|---|
| Constant base rate | 0.076 | 0.5 |
| Logistic regression, L2 | 0.177 | 0.721 |
| `HistGradientBoostingClassifier` | 0.192 | 0.738 |

### Conclusion

The model is chosen and trained well: time-based split, tuning without the test set, the test set is used
once, and two different model families give close results. The result is not outstanding: PR AUC is about
2.5 times the base rate, ROC AUC is 0.74, and the gain over the logistic regression is small.

Feature engineering from the existing columns had no effect: six derived features (price per hour, price
per battery, deposit to price ratio, number of previous late returns, planned return hour and day of week)
gave no gain in PR AUC over the 4 time-based folds on train + validation (0.230 vs 0.232 without them).

The tuning picked trees with 2 leaves, so the signal is close to additive and further tuning or another
model is unlikely to help. New informative features known at checkout time, not recombinations of the
existing columns, are likely needed to improve the model (a hypothesis, not tested).

## Running

Tested on Python 3.14.3.

1. Create a virtual environment in the project root:

   ```bash
   python -m venv .venv
   ```

2. Install the packages from `requirements.txt` (ipykernel, numpy, pandas, matplotlib, seaborn,
   scikit-learn, scipy):

   ```bash
   # Windows
   .venv\Scripts\python -m pip install -r requirements.txt
   # Linux / macOS
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Open `main.ipynb` in VS Code and select the `.venv` kernel (top right, "Select Kernel").
   Without VS Code: `.venv\Scripts\python -m pip install jupyterlab`, then `.venv\Scripts\jupyter lab`.

The notebook downloads `dataset.csv` from Google Drive, so an internet connection is required. Offline,
replace the link with the local path `"dataset.csv"`.
