import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


file_id = "1jVzIRp6zxrGVDXRtuG-8mc6CzHrbQFED"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df = pd.read_csv(
    url,
    parse_dates=["started_at"],
    dtype={"y_overdue": "boolean", "prev_late": "boolean"},
)

TARGET = "y_overdue"
NUM_COLS = ["hour", "month", "plan_hours", "n_batteries", "deposit_amount", "price_total",
            "prepaid_share", "client_prev_rentals", "client_prev_late_rate", "client_prev_lost",
            "days_since_prev"]
BOOL_COLS = ["is_weekend", "block_missing", "prev_late"]
CAT_COLS = ["tenant", "dow", "model", "deposit_kind", "client_residency"]

FEATURES = CAT_COLS + NUM_COLS + BOOL_COLS

df = df.dropna(subset=[TARGET]).reset_index(drop=True)   # lost batteries have no y_overdue
df[TARGET] = df[TARGET].astype("int8")
df[BOOL_COLS] = df[BOOL_COLS].astype("float32")          # pd.NA -> NaN

# train-test split is time-based, so we can use the same split for all models

df = df.sort_values("started_at").reset_index(drop=True)

df = df[FEATURES + [TARGET]]

df_full_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, shuffle=False)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train[TARGET].values
y_val = df_val[TARGET].values
y_test = df_test[TARGET].values

del df_train[TARGET]
del df_val[TARGET]
del df_test[TARGET]


# tuning round 2: best row by PR-AUC over 4 time-based folds on train + val
BEST_PARAMS = dict(learning_rate=0.0787, l2_regularization=10.8943, max_features=0.5226,
                   max_leaf_nodes=2, min_samples_leaf=69, max_iter=560)

def make_transformer():
    # the model handles NaN itself; categories only need integer codes (unseen category -> NaN)
    # columns are picked by name, so keys / extra columns in df are ignored
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT_COLS),
        ("rest", "passthrough", NUM_COLS + BOOL_COLS),
    ])

def make_model():
    cat_mask = [True] * len(CAT_COLS) + [False] * (len(NUM_COLS) + len(BOOL_COLS))   # same order as the transformer output
    return HistGradientBoostingClassifier(
        **BEST_PARAMS, early_stopping=False, categorical_features=cat_mask, random_state=42,
    )

pipe = Pipeline([
    ("prep", make_transformer()),
    ("model", make_model()),
])

pipe.fit(df_train, y_train)

p_test = pipe.predict_proba(df_test)[:, 1]

print("base   ", y_test.mean())
print("PR-AUC ", average_precision_score(y_test, p_test))
print("ROC AUC", roc_auc_score(y_test, p_test))

output_file = "model_hgb.bin"
with open(output_file, "wb") as f_out:
    pickle.dump(pipe, f_out)