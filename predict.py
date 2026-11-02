import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

with open("model_hgb.bin", "rb") as f_in:
    pipe = pickle.load(f_in)

app = FastAPI()

# test set: p >= HIGH flags 8% of checkouts and 24% of them are overdue (base rate 7.6%)
HIGH = 0.20
LOW = 0.08


def risk(p):
    if p >= HIGH:
        return "high", f"Warning: high risk, {p:.1%} chance of an overdue return (average is about 8%)."
    if p < LOW:
        return "low", f"Low risk: {p:.1%} chance of an overdue return (average is about 8%)."
    return "medium", f"Moderate risk: {p:.1%} chance of an overdue return (average is about 8%)."


@app.post("/predict")
def predict(row: dict):
    missing = set(pipe.feature_names_in_) - row.keys()
    if missing:
        raise HTTPException(422, f"missing fields: {sorted(missing)}")
    df = pd.DataFrame([row])
    p = float(pipe.predict_proba(df)[0, 1])
    level, message = risk(p)
    return {"overdue_probability": p, "risk": level, "message": message}