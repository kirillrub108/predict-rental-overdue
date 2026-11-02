import json
import pickle
import pandas as pd

with open("model_hgb.bin", "rb") as f_in:
    pipe = pickle.load(f_in)

def predict(model, df):
    return model.predict_proba(df)[:, 1]

with open("test_request.json", encoding="utf-8") as f:
    df_new = pd.DataFrame([json.load(f)])    # одна строка

y_pred = predict(pipe, df_new)
print(y_pred[0])