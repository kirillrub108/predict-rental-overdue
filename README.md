# Midterm Project — forecasting overdue rental checkouts

The goal is to predict, at the moment a bike is handed out, whether it will come back late:
more than 15 minutes after the planned return time (`y_overdue`).

| Path | What it is |
|---|---|
| `main.ipynb` | notebook: data loading, EDA, model |
| `dataset.csv` | ready-made dataset: one row is one checkout, features known at checkout time, targets `y_overdue`, `y_lost` |

The data is synthetic: the signal is baked in by the generator, so the model's metrics do not carry over to real data.

## Target

Only overdue returns are modelled. Lost bikes (`y_lost`, ~1% of checkouts, ~390 cases) have no overdue
label and are excluded: they are too rare for a model of their own and in practice are handled by the
deposit policy. A loss costs about as much as a hundred late returns, so mixing both into one target
would be misleading.

## EDA (in `main.ipynb`)

Overdue-only, 38,953 checkouts, overdue rate 8.29%:

1. Data quality and value ranges
2. Target: class balance, rate by tenant
3. Missing values: random gaps in the form vs informative gaps in client history
4. Feature distributions (including a 09:00 spike — a generator artifact)
5. Feature vs target: overdue rate by group with ±2 standard errors
6. Relations between features: price ≈ duration, deposit vs residency
7. Time: seasonality, stability of the rate, why the split must be by time
8. Single-feature importance: mutual information and ROC-AUC
9. Leakage check
10. Conclusions → decisions for the model

## Running

Requires Python 3.12 or newer (tested on 3.14).

1. Create a virtual environment in the project root:

   ```bash
   python -m venv .venv
   ```

2. Install the packages from `requirements.txt` (pandas, matplotlib, scikit-learn, ipykernel):

   ```bash
   # Windows
   .venv\Scripts\python -m pip install -r requirements.txt
   # Linux / macOS
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Open `main.ipynb` in VS Code and select the `.venv` kernel (top right, "Select Kernel").
   Without VS Code: `.venv\Scripts\python -m pip install jupyterlab`, then `.venv\Scripts\jupyter lab`.

The notebook downloads `dataset.csv` from Google Drive, so an internet connection is required. Offline, replace the link with
the local path `"dataset.csv"`.
