"""
FINAL FIXED main.py — NO FILE RENAMING REQUIRED
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Delhi Weather Predictor", layout="wide")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
base = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
m = {}

```
def load(name):
    try:
        return joblib.load(os.path.join(base, "models", name))
    except:
        return None

m["xgb"] = load("xgboost_model.pkl")
m["lgb"] = load("lightgbm_model.pkl")
m["arima"] = load("arima_model.pkl")
m["sarima"] = load("sarima_model.pkl")
m["meta"] = load("feature_meta.pkl")
m["lstm_scaler"] = load("lstm_scaler.pkl")

try:
    import tensorflow as tf
    m["lstm"] = tf.keras.models.load_model(
        os.path.join(base, "models/lstm_model.keras")
    )
except:
    m["lstm"] = None

return m
```

models = load_models()

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def build_input(temp, hum, wind, press, t1, t2, features):
row = {f: 0.0 for f in features}
row.update({
"meantemp": temp,
"humidity": hum,
"wind_speed": wind,
"meanpressure": press,
"temp_lag1": t1,
"temp_lag2": t2,
})
return pd.DataFrame([row]).reindex(columns=features, fill_value=0)

# ── ARIMA ─────────────────────────────────────────────────────────────────────

def arima_step(model, t1):
try:
return float(model.append([t1], refit=False).forecast(1)[0])
except:
return None

# ── ✅ FIXED LSTM ─────────────────────────────────────────────────────────────

def lstm_step(model, scaler, temp, hum, wind, press, t1, t2):
try:
LOOKBACK = 30
n_features = scaler.n_features_in_

```
    temps = np.linspace(t2, temp, LOOKBACK)

    window = np.column_stack([
        temps,
        np.full(LOOKBACK, hum),
        np.full(LOOKBACK, wind),
        np.full(LOOKBACK, press),
    ])

    if window.shape[1] < n_features:
        window = np.hstack([
            window,
            np.zeros((LOOKBACK, n_features - window.shape[1]))
        ])

    X = scaler.transform(window).reshape(1, LOOKBACK, n_features)
    y = model.predict(X, verbose=0)[0][0]

    dummy = np.zeros((1, n_features))
    dummy[0, 0] = y

    return float(scaler.inverse_transform(dummy)[0, 0])
except:
    return None
```

# ── ENSEMBLE ──────────────────────────────────────────────────────────────────

RMSE = {
"XGB": 0.49,
"LGB": 0.44,
"LSTM": 1.99,
"ARIMA": 1.74,
"SARIMA": 1.73
}

def ensemble(preds):
w = {k: 1/v for k, v in RMSE.items() if k in preds}
s = sum(w.values())
return sum(preds[k]*w[k]/s for k in w)

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌤️ Delhi Weather Predictor")

temp = st.slider("Temperature (°C)", 0.0, 45.0, 22.0)
hum  = st.slider("Humidity (%)", 10.0, 100.0, 65.0)
wind = st.slider("Wind Speed", 0.0, 50.0, 8.0)
press= st.slider("Pressure", 990.0, 1030.0, 1010.0)

t1 = st.number_input("Yesterday Temp", value=21.0)
t2 = st.number_input("2 Days Ago Temp", value=20.0)

if st.button("Predict"):
preds = {}

```
if models["xgb"]:
    X = build_input(temp, hum, wind, press, t1, t2,
                    models["meta"]["features"])
    preds["XGB"] = float(models["xgb"].predict(X)[0])

if models["lgb"]:
    preds["LGB"] = float(models["lgb"].predict(X)[0])

if models["lstm"] and models["lstm_scaler"]:
    preds["LSTM"] = lstm_step(
        models["lstm"], models["lstm_scaler"],
        temp, hum, wind, press, t1, t2
    )

if models["arima"]:
    preds["ARIMA"] = arima_step(models["arima"], t1)

if models["sarima"]:
    preds["SARIMA"] = arima_step(models["sarima"], t1)

st.subheader("Predictions")
for k, v in preds.items():
    st.write(f"{k}: {v:.2f} °C")

if preds:
    st.success(f"🌟 Ensemble: {ensemble(preds):.2f} °C")
```
