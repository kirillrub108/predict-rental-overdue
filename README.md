# Midterm Project — forecasting overdue bike rental returns

The goal is to predict, at the moment a bike is handed out, whether it will come back late:
more than 15 minutes after the planned return time (`y_overdue`).

**Status:** exploratory data analysis is finished (`main.ipynb`, ending with a General Conclusion);
no model has been trained yet.

| Path | What it is |
|---|---|
| `main.ipynb` | notebook: data loading, EDA, conclusions |
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

- Use PR AUC as the main metric and stratified splits; validate on a time-based split.
- The strongest signals are `plan_hours` and `price_total`, then `deposit_kind`, `client_residency`,
  `prepaid_share`, `first_visit`, weekends and the evening hours.
- Missing client history is informative (10.5% overdue vs 6.8%), gaps in the card block look random.
- No leakage found; `client_prev_lost`, `issued_by_employee_id` and one of `first_visit` /
  `client_prev_rentals` should be dropped as constant or redundant.

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
