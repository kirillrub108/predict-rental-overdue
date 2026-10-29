# Midterm Project — forecasting overdue rental checkouts

| Path | What it is |
|---|---|
| `main.ipynb` | notebook: data loading, EDA, model |
| `dataset.csv` | ready-made dataset: one row is one checkout, targets `y_overdue`, `y_lost` |

The data is synthetic: the signal is baked in by the generator, so the model's metrics do not carry over to real data.

## Running

Requires Python 3.12 or newer (tested on 3.14).

1. Create a virtual environment in the project root:

   ```bash
   python -m venv .venv
   ```

2. Install the packages from `requirements.txt`:

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
