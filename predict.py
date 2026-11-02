import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

with open("model_hgb.bin", "rb") as f_in:
    pipe = pickle.load(f_in)

app = FastAPI()


@app.post("/predict")
def predict(row: dict):
    missing = set(pipe.feature_names_in_) - row.keys()
    if missing:
        raise HTTPException(422, f"missing fields: {sorted(missing)}")
    df = pd.DataFrame([row])
    return {"overdue_probability": float(pipe.predict_proba(df)[0, 1])}