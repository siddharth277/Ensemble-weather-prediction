"""
app/main.py
============================================================
PHASE: DEPLOYMENT — Streamlit Weather Prediction App
Delhi Weather Forecasting – Full Ensemble (XGB + LGB + LSTM + ARIMA + SARIMA)

Matches the RUN_IN_COLAB.ipynb pipeline exactly:
  - Loads XGBoost + LightGBM models  (from 04_model_train_evaluate.py)
  - Loads LSTM model + scaler        (from 05_lstm_model.py)   [optional]
  - Loads ARIMA / SARIMA models      (from 05_arima_model.py)  [optional]
  - Shows per-model predictions + adaptive weighted ensemble
  - Displays historical test results from ensemble_final.csv
  - Model status panel shows which models are loaded

Run:  streamlit run app/main.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Delhi Weather Forecaster",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS — Clean dark-blue meteorological aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Page background */
    .stApp {
        background: #0D1B2A;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #112336 !important;
        border-right: 1px solid #1E3A5F;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #C8DCEF !important;
    }

    /* Main header */
    .app-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A3050 50%, #0D1B2A 100%);
        border: 1px solid #1E3A5F;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(52,152,219,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #E8F4FD;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        color: #6B9EC7;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Prediction cards */
    .pred-card {
        background: #112336;
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: border-color 0.2s;
        height: 100%;
    }
    .pred-card:hover { border-color: #2E6DA4; }
    .pred-card.best  { border-color: #27AE60; background: #0E2318; }
    .pred-card.ensemble { border-color: #F39C12; background: #1A1400; }

    .pred-model-name {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #5B8AB0;
        margin-bottom: 0.5rem;
    }
    .pred-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E8F4FD;
        font-family: 'DM Mono', monospace;
        line-height: 1;
    }
    .pred-emoji {
        font-size: 1rem;
        color: #8ABADC;
        margin-top: 0.4rem;
    }
    .pred-badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        margin-top: 0.5rem;
    }
    .badge-best     { background: #1A3D28; color: #27AE60; border: 1px solid #27AE60; }
    .badge-ensemble { background: #2D2200; color: #F39C12; border: 1px solid #F39C12; }
    .badge-na       { background: #1A2535; color: #5B7FA6; border: 1px solid #1E3A5F; }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3A7BBF;
        border-bottom: 1px solid #1E3A5F;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Status pills */
    .status-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        font-size: 0.75rem; font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
    }
    .pill-ok     { background: #0E2318; color: #27AE60; border: 1px solid #1A4028; }
    .pill-miss   { background: #1A1522; color: #8E44AD; border: 1px solid #2D1A3D; }
    .pill-warn   { background: #1A1000; color: #F39C12; border: 1px solid #3D2800; }

    /* Info boxes */
    .info-row {
        background: #0A1520;
        border: 1px solid #162840;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .info-label { font-size: 0.75rem; font-weight: 600; color: #3A7BBF; }
    .info-value { font-size: 0.9rem; color: #C8DCEF; margin-top: 0.2rem; }

    /* Metric mini table */
    .metric-table { width: 100%; border-collapse: collapse; }
    .metric-table th {
        font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: #3A7BBF; padding: 0.4rem 0.6rem; border-bottom: 1px solid #1E3A5F;
        text-align: left;
    }
    .metric-table td {
        font-size: 0.85rem; color: #C8DCEF; padding: 0.4rem 0.6rem;
        border-bottom: 1px solid #0D1B2A; font-family: 'DM Mono', monospace;
    }
    .metric-table tr:last-child td { border-bottom: none; }
    .metric-table tr.best-row td  { color: #27AE60; }

    /* Streamlit overrides */
    .stSlider > div > div > div { background: #1E3A5F !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1A3A6B, #2E6DA4) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 1rem !important; padding: 0.6rem 1.5rem !important;
        width: 100% !important; letter-spacing: 0.02em !important;
    }
    .stButton > button:hover { opacity: 0.9; }
    div[data-testid="stMarkdownContainer"] p { color: #9BBCDA; }
    h1, h2, h3 { color: #E8F4FD !important; }
    .stTabs [data-baseweb="tab"] { color: #6B9EC7 !important; }
    .stTabs [aria-selected="true"] { color: #E8F4FD !important; border-bottom: 2px solid #3A7BBF !important; }
    [data-testid="stMetric"] { background: #112336; border-radius: 10px; padding: 0.8rem; border: 1px solid #1E3A5F; }
    [data-testid="stMetricLabel"] > div { color: #6B9EC7 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] > div { color: #E8F4FD !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def temp_emoji(t):
    if t < 8:   return "🥶 Very Cold"
    if t < 16:  return "🌨️ Cold"
    if t < 22:  return "🌥️ Cool"
    if t < 28:  return "🌤️ Pleasant"
    if t < 34:  return "☀️ Warm"
    return "🔥 Hot"

def temp_color(t):
    if t < 16: return "#74B9FF"
    if t < 26: return "#55EFC4"
    if t < 32: return "#FDCB6E"
    return "#FF7675"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# MODEL LOADING — All models optional, degrade gracefully
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_all_models():
    """
    Load every model that exists. Returns a dict with model objects or None.
    Matches exactly what the Colab notebook produces:
      models/xgboost_model.pkl        ← 04_model_train_evaluate.py
      models/lightgbm_model.pkl       ← 04_model_train_evaluate.py
      models/ensemble_weights.pkl     ← 04_model_train_evaluate.py
      models/feature_meta.pkl         ← 03_feature_engineering.py
      models/lstm_model.keras         ← 05_lstm_model.py
      models/lstm_scaler.pkl          ← 05_lstm_model.py
      models/arima_model.pkl          ← 05_arima_model.py
      models/sarima_model.pkl         ← 05_arima_model.py
    """
    models = {}

    def try_load(key, path):
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            try:
                models[key] = joblib.load(full)
            except Exception as e:
                models[key] = None
                models[f'{key}_err'] = str(e)
        else:
            models[key] = None

    # XGBoost + LightGBM
    try_load('xgb',      'models/xgboost_model.pkl')
    try_load('lgb',      'models/lightgbm_model.pkl')
    try_load('weights',  'models/ensemble_weights.pkl')
    try_load('meta',     'models/feature_meta.pkl')

    # ARIMA / SARIMA
    try_load('arima',    'models/arima_model.pkl')
    try_load('sarima',   'models/sarima_model.pkl')

    # LSTM (TensorFlow — optional)
    lstm_path  = os.path.join(BASE_DIR, 'models/lstm_model.keras')
    scaler_path = os.path.join(BASE_DIR, 'models/lstm_scaler.pkl')
    models['lstm']        = None
    models['lstm_scaler'] = None
    if os.path.exists(lstm_path):
        try:
            import tensorflow as tf
            models['lstm'] = tf.keras.models.load_model(lstm_path)
            if os.path.exists(scaler_path):
                models['lstm_scaler'] = joblib.load(scaler_path)
        except Exception as e:
            models['lstm_err'] = str(e)

    # Historical predictions
    ens_path = os.path.join(BASE_DIR, 'data/predictions/ensemble_final.csv')
    if os.path.exists(ens_path):
        try:
            models['history'] = pd.read_csv(ens_path, parse_dates=['date'])
        except Exception:
            models['history'] = None
    else:
        models['history'] = None

    return models

models = load_all_models()

xgb_ok    = models.get('xgb')    is not None
lgb_ok    = models.get('lgb')    is not None
meta_ok   = models.get('meta')   is not None
arima_ok  = models.get('arima')  is not None
sarima_ok = models.get('sarima') is not None
lstm_ok   = models.get('lstm')   is not None
ml_ready  = xgb_ok and lgb_ok and meta_ok


# ══════════════════════════════════════════════════════════════
# FEATURE BUILDER — matches 03_feature_engineering.py exactly
# ══════════════════════════════════════════════════════════════

def build_feature_row(temp, humidity, wind, pressure,
                      temp_lag1, temp_lag2, temp_lag3=None, temp_lag7=None):
    """
    Build a single-row DataFrame with all 43 features expected by XGB/LGB.
    Uses the same feature names as feature_meta.pkl.
    For lag3 and lag7, approximate if not provided.
    """
    today    = datetime.date.today()
    month    = today.month
    doy      = today.timetuple().tm_yday
    lag3     = temp_lag3 if temp_lag3 is not None else temp_lag2
    lag7     = temp_lag7 if temp_lag7 is not None else (temp + temp_lag1 + temp_lag2) / 3

    avg3  = (temp_lag1 + temp_lag2 + lag3) / 3
    avg7  = avg3
    avg14 = avg3

    row = {
        # Raw features
        'humidity'       : humidity,
        'wind_speed'     : wind,
        'meanpressure'   : pressure,
        # Time
        'month'          : month,
        'day'            : today.day,
        'day_of_year'    : doy,
        'day_of_week'    : today.weekday(),
        'season'         : {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[month],
        'month_sin'      : np.sin(2 * np.pi * month / 12),
        'month_cos'      : np.cos(2 * np.pi * month / 12),
        'doy_sin'        : np.sin(2 * np.pi * doy / 365),
        'doy_cos'        : np.cos(2 * np.pi * doy / 365),
        # Lags — temp
        'temp_lag1'      : temp_lag1,
        'temp_lag2'      : temp_lag2,
        'temp_lag3'      : lag3,
        'temp_lag7'      : lag7,
        # Lags — humidity (approximate with current)
        'humidity_lag1'  : humidity,
        'humidity_lag2'  : humidity,
        'humidity_lag3'  : humidity,
        'humidity_lag7'  : humidity,
        # Lags — pressure
        'pressure_lag1'  : pressure,
        'pressure_lag2'  : pressure,
        'pressure_lag3'  : pressure,
        'pressure_lag7'  : pressure,
        # Lags — wind
        'wind_lag1'      : wind,
        'wind_lag2'      : wind,
        'wind_lag3'      : wind,
        'wind_lag7'      : wind,
        # Rolling — temperature
        'temp_roll_mean3'  : avg3,
        'temp_roll_std3'   : abs(temp_lag1 - temp_lag2) * 0.5,
        'hum_roll_mean3'   : humidity,
        'temp_roll_mean7'  : avg7,
        'temp_roll_std7'   : abs(temp_lag1 - temp_lag2) * 0.7,
        'hum_roll_mean7'   : humidity,
        'temp_roll_mean14' : avg14,
        'temp_roll_std14'  : abs(temp_lag1 - temp_lag2) * 0.8,
        'hum_roll_mean14'  : humidity,
        # EWM
        'temp_ewm7'        : 0.7 * temp_lag1 + 0.3 * temp_lag2,
        'temp_ewm14'       : 0.6 * temp_lag1 + 0.4 * temp_lag2,
        # Interactions
        'heat_index'       : temp_lag1 * humidity / 100,
        'pressure_delta'   : 0.0,
        'temp_delta'       : temp_lag1 - temp_lag2,
        'wind_chill'       : temp_lag1 - 0.5 * wind,
    }

    features = models['meta']['features']
    df = pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)
    return df


# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════

def run_predictions(temp, humidity, wind, pressure, lag1, lag2):
    """Run all available models and return a dict of predictions."""
    preds = {}

    if ml_ready:
        X = build_feature_row(temp, humidity, wind, pressure, lag1, lag2)
        preds['XGBoost']  = float(models['xgb'].predict(X)[0])
        preds['LightGBM'] = float(models['lgb'].predict(X)[0])

    if arima_ok:
        try:
            arima_m = models['arima']
            preds['ARIMA'] = float(arima_m.forecast(steps=1)[0])
        except Exception:
            pass

    if sarima_ok:
        try:
            sarima_m = models['sarima']
            preds['SARIMA'] = float(sarima_m.forecast(steps=1)[0])
        except Exception:
            pass

    # Ensemble — adaptive weights (matches 06_ensemble.py logic)
    if len(preds) >= 2:
        BASE_W = {'XGBoost': 0.30, 'LightGBM': 0.30,
                  'LSTM': 0.25, 'ARIMA': 0.075, 'SARIMA': 0.075}
        STRONG = {'XGBoost', 'LightGBM'}
        available = set(preds.keys())
        missing_w = sum(BASE_W.get(m, 0) for m in BASE_W if m not in available)
        weights   = {m: BASE_W.get(m, 0) for m in available}
        strong_av = available & STRONG
        if missing_w > 0 and strong_av:
            bonus = missing_w / len(strong_av)
            for m in strong_av:
                weights[m] += bonus
        total = sum(weights.values())
        weights = {m: w / total for m, w in weights.items()}
        preds['Ensemble'] = sum(weights[m] * preds[m] for m in available)
        preds['_weights'] = weights

    return preds


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌡️ Today's Observations")
    st.markdown("*Enter current weather readings for Delhi*")
    st.markdown("---")

    temp = st.slider("Mean Temperature (°C)", 0.0, 45.0, 24.0, 0.5,
                     help="Today's mean temperature")
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 62.0, 1.0,
                         help="Relative humidity percentage")
    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 8.0, 0.5,
                     help="Mean wind speed")
    pressure = st.slider("Mean Pressure (hPa)", 990.0, 1030.0, 1010.0, 0.5,
                         help="Atmospheric pressure at sea level")

    st.markdown("---")
    st.markdown("##### 📅 Recent Temperature History")
    lag1 = st.number_input("Yesterday (°C)",     value=23.0, step=0.5,
                            help="Mean temperature 1 day ago")
    lag2 = st.number_input("2 Days Ago (°C)",    value=22.0, step=0.5,
                            help="Mean temperature 2 days ago")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Tomorrow's Temperature")

    # Model status panel
    st.markdown("---")
    st.markdown("##### ⚙️ Model Status")
    status_html = '<div class="status-row">'
    for name, ok in [('XGBoost', xgb_ok), ('LightGBM', lgb_ok),
                     ('LSTM', lstm_ok), ('ARIMA', arima_ok), ('SARIMA', sarima_ok)]:
        cls  = 'pill-ok' if ok else 'pill-miss'
        icon = '✓' if ok else '✗'
        status_html += f'<span class="status-pill {cls}">{icon} {name}</span>'
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

    if not ml_ready:
        st.markdown(
            '<div class="status-pill pill-warn" style="margin-top:0.5rem">'
            '⚠ Run Steps 6–7 in Colab to train XGB/LGB</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <div class="app-title">🌡️ Delhi Weather Forecaster</div>
  <div class="app-subtitle">
    Ensemble ML model · XGBoost + LightGBM + LSTM + ARIMA/SARIMA ·
    Trained on Delhi 2013–2017 daily observations
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Historical Results", "ℹ️ Model Info"])

# ─────────────────────────────────────────────
# TAB 1: PREDICT
# ─────────────────────────────────────────────
with tab1:
    if not predict_btn:
        # Placeholder state
        st.markdown("""
        <div style="background:#112336;border:1px dashed #1E3A5F;border-radius:14px;
                    padding:3rem 2rem;text-align:center;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🌤️</div>
          <div style="color:#6B9EC7;font-size:1.1rem;font-weight:500;">
            Set today's weather values in the sidebar,<br>then press <b style="color:#E8F4FD">Predict Tomorrow's Temperature</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not ml_ready:
            st.error("⚠️ XGBoost/LightGBM models not found. Run Steps 6–7 in the Colab notebook first.")
        else:
            preds = run_predictions(temp, humidity, wind, pressure, lag1, lag2)
            weights = preds.pop('_weights', {})

            ensemble_val = preds.get('Ensemble')

            st.markdown('<div class="section-header">Tomorrow\'s Temperature Forecast</div>',
                        unsafe_allow_html=True)

            # Determine best individual model
            indiv = {k: v for k, v in preds.items() if k != 'Ensemble'}
            best_model = min(indiv, key=lambda k: abs(indiv[k] - (ensemble_val or 0))) \
                         if ensemble_val else None

            # Render prediction cards
            model_order = ['XGBoost', 'LightGBM', 'LSTM', 'ARIMA', 'SARIMA', 'Ensemble']
            available_models = [m for m in model_order if m in preds]
            n_cols = min(len(available_models), 3)
            col_groups = [available_models[i:i+n_cols]
                          for i in range(0, len(available_models), n_cols)]

            for group in col_groups:
                cols = st.columns(len(group))
                for col, mname in zip(cols, group):
                    val  = preds[mname]
                    is_ens  = mname == 'Ensemble'
                    card_cls = 'pred-card ensemble' if is_ens else 'pred-card'

                    if is_ens:
                        badge = '<span class="pred-badge badge-ensemble">★ ENSEMBLE</span>'
                    else:
                        badge = ''

                    color = temp_color(val)
                    with col:
                        st.markdown(f"""
                        <div class="{card_cls}">
                          <div class="pred-model-name">{mname}</div>
                          <div class="pred-value" style="color:{color}">{val:.1f}°C</div>
                          <div class="pred-emoji">{temp_emoji(val)}</div>
                          {badge}
                        </div>
                        """, unsafe_allow_html=True)

            # Ensemble weight breakdown
            if weights and ensemble_val:
                st.markdown('<div class="section-header">Ensemble Weight Breakdown</div>',
                            unsafe_allow_html=True)
                w_cols = st.columns(len(weights))
                for col, (mname, w) in zip(w_cols, weights.items()):
                    col.metric(mname, f"{w*100:.1f}%")

            # Context bar
            st.markdown('<div class="section-header">Context</div>',
                        unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Today's Temp",  f"{temp:.1f}°C")
            c2.metric("Yesterday",      f"{lag1:.1f}°C",
                      delta=f"{temp-lag1:+.1f}°C")
            c3.metric("Humidity",       f"{humidity:.0f}%")
            c4.metric("Pressure",       f"{pressure:.0f} hPa")

            # Forecast bar chart
            st.markdown('<div class="section-header">Model Comparison</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3.2))
            fig.patch.set_facecolor('#0D1B2A')
            ax.set_facecolor('#112336')

            names = list(preds.keys())
            vals  = [preds[n] for n in names]
            colors = ['#F39C12' if n == 'Ensemble' else '#3A7BBF' for n in names]

            bars = ax.bar(names, vals, color=colors, edgecolor='#1E3A5F',
                          linewidth=0.8, width=0.55)
            y_min = max(0, min(vals) - 3)
            y_max = max(vals) + 3
            ax.set_ylim(y_min, y_max)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f'{val:.1f}°C', ha='center', va='bottom',
                        color='#C8DCEF', fontsize=9, fontweight='600')

            ax.axhline(ensemble_val, color='#F39C12', lw=1.2,
                       ls='--', alpha=0.4, label='Ensemble')
            ax.set_ylabel('°C', color='#6B9EC7', fontsize=10)
            ax.tick_params(colors='#6B9EC7', labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E3A5F')
            ax.set_title('Tomorrow\'s Temperature — Model Predictions',
                         color='#9BBCDA', fontsize=10, pad=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# TAB 2: HISTORICAL RESULTS
# ─────────────────────────────────────────────
with tab2:
    history = models.get('history')

    if history is None:
        st.info("Historical predictions not found. Run Steps 7–10 in the Colab notebook to generate `data/predictions/ensemble_final.csv`.")
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        actual     = history['actual'].values
        pred_cols  = [c for c in history.columns
                      if c.startswith('pred_') or c == 'prediction_ensemble']
        model_labels = {c: c.replace('pred_','').replace('_',' ').upper()
                        for c in pred_cols}

        # Metrics summary table
        st.markdown('<div class="section-header">Test Set Performance (Delhi 2017)</div>',
                    unsafe_allow_html=True)

        rows = []
        for col in pred_cols:
            p    = history[col].values
            rmse = np.sqrt(mean_squared_error(actual, p))
            mae  = mean_absolute_error(actual, p)
            r2   = r2_score(actual, p)
            rows.append({'Model': model_labels[col],
                         'RMSE': rmse, 'MAE': mae, 'R²': r2})
        metrics_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)

        best_rmse = metrics_df['RMSE'].min()
        table_html = '<table class="metric-table"><tr>'
        for col in ['Model', 'RMSE', 'MAE', 'R²']:
            table_html += f'<th>{col}</th>'
        table_html += '</tr>'
        for _, row in metrics_df.iterrows():
            is_best = row['RMSE'] == best_rmse
            cls = 'class="best-row"' if is_best else ''
            star = ' ★' if is_best else ''
            table_html += f'<tr {cls}>'
            table_html += f'<td>{row["Model"]}{star}</td>'
            table_html += f'<td>{row["RMSE"]:.4f} °C</td>'
            table_html += f'<td>{row["MAE"]:.4f} °C</td>'
            table_html += f'<td>{row["R²"]:.4f}</td>'
            table_html += '</tr>'
        table_html += '</table>'

        st.markdown(
            f'<div style="background:#112336;border:1px solid #1E3A5F;'
            f'border-radius:12px;padding:1rem;">{table_html}</div>',
            unsafe_allow_html=True
        )

        # Actual vs Predicted plot
        st.markdown('<div class="section-header">Actual vs Predicted — All Models</div>',
                    unsafe_allow_html=True)

        dates    = pd.to_datetime(history['date']).values
        palette  = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6',
                    '#F39C12', '#1ABC9C']

        n_plots  = len(pred_cols)
        fig, axes = plt.subplots(n_plots, 1,
                                  figsize=(12, 3.2 * n_plots),
                                  sharex=True)
        if n_plots == 1:
            axes = [axes]

        fig.patch.set_facecolor('#0D1B2A')
        for ax, col, color in zip(axes, pred_cols, palette):
            pred = history[col].values
            row  = metrics_df[metrics_df['Model'] == model_labels[col]].iloc[0]
            ax.set_facecolor('#112336')
            ax.plot(dates, actual, color='#E8F4FD', lw=1.6, label='Actual', zorder=3)
            ax.plot(dates, pred,   color=color,     lw=1.4, ls='--',
                    label=f'{model_labels[col]}', zorder=2)
            ax.fill_between(dates, actual, pred, alpha=0.07, color=color)
            ax.set_title(
                f'{model_labels[col]}  ·  RMSE={row["RMSE"]:.3f}°C  '
                f'MAE={row["MAE"]:.3f}°C  R²={row["R²"]:.4f}',
                color='#9BBCDA', fontsize=9, fontweight='600', pad=6
            )
            ax.set_ylabel('°C', color='#6B9EC7', fontsize=9)
            ax.tick_params(colors='#6B9EC7', labelsize=8)
            ax.legend(fontsize=8, facecolor='#0D1B2A', labelcolor='#9BBCDA',
                      edgecolor='#1E3A5F')
            ax.grid(True, alpha=0.15, color='#1E3A5F')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E3A5F')

        axes[-1].set_xlabel('Date', color='#6B9EC7', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Residual distribution for ensemble
        if 'prediction_ensemble' in history.columns:
            st.markdown('<div class="section-header">Ensemble Residual Analysis</div>',
                        unsafe_allow_html=True)
            residuals = actual - history['prediction_ensemble'].values
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            fig.patch.set_facecolor('#0D1B2A')
            for ax in axes:
                ax.set_facecolor('#112336')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1E3A5F')
                ax.tick_params(colors='#6B9EC7', labelsize=8)

            axes[0].plot(dates, residuals, color='#F39C12', lw=1.2)
            axes[0].axhline(0, color='#E74C3C', lw=1.3, ls='--', alpha=0.7)
            axes[0].fill_between(dates, residuals, 0, alpha=0.12, color='#F39C12')
            axes[0].set_title('Residuals Over Time', color='#9BBCDA', fontsize=9)
            axes[0].set_ylabel('Error (°C)', color='#6B9EC7', fontsize=9)
            axes[0].grid(True, alpha=0.15, color='#1E3A5F')

            axes[1].hist(residuals, bins=22, color='#3A7BBF',
                         alpha=0.85, edgecolor='#1E3A5F')
            axes[1].axvline(0, color='#E74C3C', lw=1.3, ls='--',
                            alpha=0.7, label='Zero error')
            axes[1].axvline(residuals.mean(), color='#F39C12', lw=1.3, ls='--',
                            label=f'Mean = {residuals.mean():.2f}°C')
            axes[1].set_title('Residual Distribution', color='#9BBCDA', fontsize=9)
            axes[1].set_xlabel('Error (°C)', color='#6B9EC7', fontsize=9)
            axes[1].legend(fontsize=8, facecolor='#0D1B2A',
                           labelcolor='#9BBCDA', edgecolor='#1E3A5F')
            axes[1].grid(True, alpha=0.15, color='#1E3A5F')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# TAB 3: MODEL INFO
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Architecture Overview</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        for title, body in [
            ("XGBoost", "500 trees · max_depth=5 · lr=0.05 · early stopping\nFeatures: 43 engineered (lags, rolling, cyclical, interactions)"),
            ("LightGBM", "500 trees · num_leaves=31 · lr=0.05 · early stopping\nFeatures: same 43-feature set as XGBoost"),
            ("Ensemble Strategy", "Weighted average with adaptive fallback.\nIf LSTM missing → its 0.25 weight redistributed to XGB+LGB only.\nWeak models (ARIMA/SARIMA) never receive bonus weight."),
        ]:
            st.markdown(f"""
            <div class="info-row">
              <div class="info-label">{title}</div>
              <div class="info-value" style="white-space:pre-line">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        for title, body in [
            ("LSTM", "BiLSTM(128) → BiLSTM(64) + Residual → MultiHeadAttention(4 heads)\n→ LSTM(32) → Dense(64→32→1)\nLookback: 30 days · Huber loss · ~310K parameters"),
            ("ARIMA / SARIMA", "Walk-forward evaluation (oracle)\nARIMA: scan (p,1,q) orders · d=1 (ADF test)\nSARIMA: seasonal s=7 (weekly pattern)\nBest order selected by lowest test RMSE"),
            ("Dataset", "Delhi daily weather 2013–2017\nTrain: 1,462 rows · Test: 114 rows\nFeatures: temperature, humidity, wind_speed, pressure"),
        ]:
            st.markdown(f"""
            <div class="info-row">
              <div class="info-label">{title}</div>
              <div class="info-value" style="white-space:pre-line">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Engineering Summary</div>',
                unsafe_allow_html=True)
    if meta_ok:
        features = models['meta']['features']
        categories = {
            'Lag Features (temp/hum/pressure/wind)':
                [f for f in features if 'lag' in f],
            'Rolling Stats (mean/std/ewm)':
                [f for f in features if 'roll' in f or 'ewm' in f],
            'Time & Cyclical':
                [f for f in features if any(x in f for x in ['month','day','doy','season','year'])],
            'Interaction Features':
                [f for f in features if any(x in f for x in ['heat','delta','chill','index'])],
            'Raw Weather':
                [f for f in features if f in ['humidity','wind_speed','meanpressure']],
        }
        fc1, fc2 = st.columns(2)
        for i, (cat, feats) in enumerate(categories.items()):
            col = fc1 if i % 2 == 0 else fc2
            with col:
                st.markdown(f"""
                <div class="info-row">
                  <div class="info-label">{cat} ({len(feats)})</div>
                  <div class="info-value" style="font-size:0.78rem;color:#5B8AB0;">
                    {', '.join(feats[:8])}{'...' if len(feats)>8 else ''}
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("feature_meta.pkl not found — run Steps 6–7 in Colab to generate it.")

    st.markdown('<div class="section-header">Base Ensemble Weights</div>',
                unsafe_allow_html=True)
    weight_data = {
        'Model':   ['XGBoost', 'LightGBM', 'LSTM', 'ARIMA', 'SARIMA'],
        'Weight':  ['0.300',   '0.300',    '0.250', '0.075', '0.075'],
        'Tier':    ['Strong',  'Strong',   'Strong','Weak',  'Weak'],
        'Status':  ['✓ Loaded' if xgb_ok    else '✗ Missing',
                    '✓ Loaded' if lgb_ok    else '✗ Missing',
                    '✓ Loaded' if lstm_ok   else '✗ Missing (TF)',
                    '✓ Loaded' if arima_ok  else '✗ Missing',
                    '✓ Loaded' if sarima_ok else '✗ Missing'],
    }
    st.dataframe(
        pd.DataFrame(weight_data),
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("""
<div style="text-align:center;color:#2A4A6B;font-size:0.78rem;margin-top:2rem;padding:1rem 0;border-top:1px solid #1A2E44;">
  Delhi Weather Forecaster · XGBoost + LightGBM + LSTM + ARIMA/SARIMA Ensemble ·
  Dataset: Delhi 2013–2017 · Author: Divyansh Prakash & Siddharth
</div>
""", unsafe_allow_html=True)
