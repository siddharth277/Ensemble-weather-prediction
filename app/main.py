"""
app/main.py
============================================================
PHASE: DEPLOYMENT — STREAMLIT WEATHER PREDICTION APP
Delhi Weather Forecasting — All Models UI
============================================================
Models supported:
  • XGBoost       (models/xgboost_model.pkl)       — always required
  • LightGBM      (models/lightgbm_model.pkl)      — always required
  • LSTM          (models/lstm_model.keras)         — optional
  • LSTM Scaler   (models/lstm_scaler.pkl)          — required for LSTM
  • ARIMA         (models/arima_model.pkl)          — optional
  • SARIMA        (models/sarima_model.pkl)         — optional
  • Ensemble      (weighted average of all loaded models)

Run:  streamlit run app/main.py

─────────────────────────────────────────────────────────────
FIX LOG (2 bugs fixed):

FIX 1 — LSTM "N/A" / not responding:
  Root cause: lstm_one_step() was creating a brand-new
  MinMaxScaler and fitting it on a synthetic window of the
  CURRENT inputs. This scaler has zero relationship to the
  one used during training, so inverse_transform produces
  garbage (often negative or >100°C) and the function
  silently returned None → UI showed "N/A".
  Fix: load models/lstm_scaler.pkl (saved during training)
  and pass it in. The Colab notebook fix is below.

FIX 2 — Ensemble too low (~26.5°C instead of ~29.5°C):
  Root cause: RMSE values were wrong placeholders
  (XGB=2.1, LGB=2.2 etc.). The actual training results are:
    LGB   RMSE=0.44  ← should dominate
    XGB   RMSE=0.49
    SARIMA RMSE=1.73
    ARIMA  RMSE=1.74
    LSTM   RMSE=1.99
  With wrong values, ARIMA/SARIMA had equal weight to XGB/LGB,
  dragging ensemble toward their ~23–24°C predictions.
  Fix: use TRUE_RMSE dict with actual training values.

COLAB NOTEBOOK FIX (add to Step 8 — notebooks/05_lstm_model.py
or the cell that saves the LSTM model):

  # After fitting the scaler, BEFORE saving the model:
  import joblib
  joblib.dump(scaler, 'models/lstm_scaler.pkl')
  print("Saved lstm_scaler.pkl")

─────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Weather Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #0F2942; letter-spacing: -1px; margin-bottom: 0.1rem; }
    .sub-header { color: #5B7FA6; font-size: 1rem; margin-bottom: 1.5rem; font-weight: 300; }
    .badge-active  { display:inline-block; background:#D1FAE5; color:#065F46; border:1px solid #6EE7B7; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-missing { display:inline-block; background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pred-card { background: linear-gradient(135deg, #0F2942 0%, #1a4a7a 100%); color: white; padding: 1.2rem 1rem; border-radius: 14px; text-align: center; margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 15px rgba(15,41,66,0.3); }
    .pred-card.ensemble  { background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%); box-shadow: 0 4px 20px rgba(124,58,237,0.4); transform: scale(1.03); }
    .pred-card.lstm-card { background: linear-gradient(135deg, #065F46 0%, #047857 100%); }
    .pred-card.arima-card  { background: linear-gradient(135deg, #7C2D12 0%, #B45309 100%); }
    .pred-card.sarima-card { background: linear-gradient(135deg, #1E3A5F 0%, #1D4ED8 100%); }
    .pred-value      { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; color:#FCD34D; line-height:1.1; }
    .pred-model-name { font-size:0.75rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; opacity:0.8; margin-bottom:0.3rem; }
    .pred-condition  { font-size:0.85rem; opacity:0.9; margin-top:0.2rem; }
    .pred-na         { font-family:'Space Mono',monospace; font-size:1.2rem; color:#FCD34D; opacity:0.5; }
    .info-box { background:#F0F7FF; border-left:3px solid #3B82F6; padding:0.7rem 1rem; border-radius:0 8px 8px 0; margin:0.4rem 0; font-size:0.9rem; }
    .info-box b { color:#1E3A5F; }
    .section-title { font-family:'Space Mono',monospace; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; color:#5B7FA6; margin:1.2rem 0 0.6rem 0; border-bottom:1px solid #E2E8F0; padding-bottom:0.3rem; }
    [data-testid="stSidebar"] { background: #0F2942; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
    .stButton > button { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; color: white !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 0.7rem 1.5rem !important; font-size: 1rem !important; width: 100% !important; letter-spacing: 0.5px; box-shadow: 0 4px 12px rgba(124,58,237,0.4) !important; }
    .note-box { background:#FFFBEB; border:1px solid #FCD34D; border-radius:8px; padding:0.6rem 0.9rem; font-size:0.82rem; color:#78350F; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load All Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = {'_errors': {}}

    for key, fname in [('xgb','xgboost_model.pkl'),
                        ('lgb','lightgbm_model.pkl'),
                        ('weights','ensemble_weights.pkl'),
                        ('meta','feature_meta.pkl'),
                        ('arima','arima_model.pkl'),
                        ('sarima','sarima_model.pkl'),
                        # FIX 1: load the scaler saved during LSTM training
                        ('lstm_scaler','lstm_scaler.pkl')]:
        try:
            m[key] = joblib.load(os.path.join(base, 'models', fname))
        except Exception as e:
            m[key] = None
            m['_errors'][key] = str(e)

    try:
        import tensorflow as tf
        m['lstm'] = tf.keras.models.load_model(os.path.join(base, 'models/lstm_model.keras'))
    except Exception as e:
        m['lstm'] = None
        m['_errors']['lstm'] = str(e)

    return m


models  = load_all_models()
core_ok = models['xgb'] is not None and models['meta'] is not None


# ── Feature Engineering (XGB / LGB) ───────────────────────────────────────────
def build_input_row(temp, humidity, wind_speed, pressure,
                    temp_yesterday, temp_2days_ago, features):
    today = datetime.date.today()
    month = today.month
    doy   = today.timetuple().tm_yday
    row   = {f: 0.0 for f in features}
    row.update({'meantemp': temp, 'humidity': humidity,
                'wind_speed': wind_speed, 'meanpressure': pressure,
                'month': month, 'day': today.day,
                'day_of_year': doy, 'day_of_week': today.weekday(),
                'season': {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[month],
                'month_sin': np.sin(2*np.pi*month/12),
                'month_cos': np.cos(2*np.pi*month/12),
                'doy_sin':   np.sin(2*np.pi*doy/365),
                'doy_cos':   np.cos(2*np.pi*doy/365),
                'temp_lag1': temp_yesterday, 'temp_lag2': temp_2days_ago,
                'temp_lag3': temp_2days_ago,
                'temp_lag7': (temp + temp_yesterday + temp_2days_ago) / 3,
                'heat_index':    temp * humidity / 100,
                'pressure_delta': 0.0,
                'temp_delta':    temp - temp_yesterday,
                'wind_chill':    temp - 0.5 * wind_speed,
                'temp_ewm7':  0.7*temp + 0.3*temp_yesterday,
                'temp_ewm14': 0.6*temp + 0.4*temp_yesterday})
    for suf in ['lag1','lag2','lag3','lag7']:
        row[f'humidity_{suf}'] = humidity
        row[f'pressure_{suf}'] = pressure
        row[f'wind_{suf}']     = wind_speed
    avg3 = (temp + temp_yesterday + temp_2days_ago) / 3
    for w in [3, 7, 14]:
        row[f'temp_roll_mean{w}'] = avg3
        row[f'temp_roll_std{w}']  = abs(temp - temp_yesterday) * 0.5
        row[f'hum_roll_mean{w}']  = humidity
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)


# ── ARIMA/SARIMA single-step ───────────────────────────────────────────────────
def arima_one_step(model, temp_yesterday):
    try:
        updated = model.append([temp_yesterday], refit=False)
        return float(updated.forecast(steps=1)[0])
    except Exception:
        return None


# ── LSTM single-step (FIX 1) ──────────────────────────────────────────────────
def lstm_one_step(lstm_model, lstm_scaler, temp, humidity, wind_speed, pressure,
                  temp_yesterday, temp_2days_ago):
    """
    FIX: use the scaler saved during training (lstm_scaler.pkl), NOT a new one.
    Creating a fresh MinMaxScaler here and fitting on a 30-point synthetic
    window gives a completely wrong scale, causing inverse_transform to return
    garbage → function returns None → UI shows N/A.
    """
    try:
        LOOKBACK = 30
        temps_w  = np.linspace(temp_2days_ago, temp, LOOKBACK)
        window   = np.column_stack([
            temps_w,
            np.full(LOOKBACK, humidity),
            np.full(LOOKBACK, wind_speed),
            np.full(LOOKBACK, pressure),
        ])
        X        = lstm_scaler.transform(window).reshape(1, LOOKBACK, 4)
        y_scaled = float(lstm_model.predict(X, verbose=0)[0, 0])
        dummy    = np.zeros((1, lstm_scaler.n_features_in_))
        dummy[0, 0] = y_scaled
        return float(lstm_scaler.inverse_transform(dummy)[0, 0])
    except Exception as e:
        if '_lstm_runtime_error' not in st.session_state:
            st.session_state['_lstm_runtime_error'] = str(e)
        return None


# ── True RMSE from your training run (FIX 2) ──────────────────────────────────
# Old code used wrong placeholders (XGB≈2.1, LGB≈2.2) that made ARIMA/SARIMA
# weigh equally with XGB/LGB, dragging ensemble to ~26.5°C.
# Your actual Colab output:  LGB=0.44  XGB=0.49  SARIMA=1.73  ARIMA=1.74  LSTM=1.99
# With correct weights LGB+XGB together carry ~80% of the ensemble weight.
TRUE_RMSE = {
    'XGBoost':  0.4938,
    'LightGBM': 0.4443,
    'LSTM':     1.9935,
    'ARIMA':    1.7421,
    'SARIMA':   1.7268,
}

def run_predictions(temp, humidity, wind_speed, pressure, temp_yesterday, temp_2days_ago):
    results = {}
    X = build_input_row(temp, humidity, wind_speed, pressure,
                        temp_yesterday, temp_2days_ago, models['meta']['features'])

    if models['xgb']:
        results['XGBoost']  = float(models['xgb'].predict(X)[0])
    if models['lgb']:
        results['LightGBM'] = float(models['lgb'].predict(X)[0])

    # FIX 1: pass the loaded scaler — don't create a new one
    if models['lstm'] and models['lstm_scaler'] is not None:
        v = lstm_one_step(models['lstm'], models['lstm_scaler'],
                          temp, humidity, wind_speed, pressure,
                          temp_yesterday, temp_2days_ago)
        if v is not None:
            results['LSTM'] = v
    elif models['lstm'] and models['lstm_scaler'] is None:
        st.session_state['_lstm_runtime_error'] = (
            "lstm_scaler.pkl missing from models/. "
            "Add joblib.dump(scaler, 'models/lstm_scaler.pkl') in Colab Step 8 and re-push."
        )

    if models['arima']:
        v = arima_one_step(models['arima'], temp_yesterday)
        if v is not None: results['ARIMA'] = v
    if models['sarima']:
        v = arima_one_step(models['sarima'], temp_yesterday)
        if v is not None: results['SARIMA'] = v

    # FIX 2: use TRUE_RMSE (inverse-RMSE weighting with real training values)
    w  = {k: 1.0 / TRUE_RMSE[k] for k in results if k in TRUE_RMSE}
    tw = sum(w.values())
    if tw > 0:
        results['Ensemble'] = sum(results[k] * w[k] / tw for k in w)
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────
def condition(t):
    if t is None: return '—'
    if t < 10:  return "🥶 Very Cold"
    if t < 18:  return "🌥️ Cool"
    if t < 26:  return "🌤️ Pleasant"
    if t < 32:  return "☀️ Warm"
    return "🔥 Hot"

CARD_CLASS  = {'LSTM':'lstm-card','ARIMA':'arima-card','SARIMA':'sarima-card','Ensemble':'ensemble'}
COLOR_MAP   = {'XGBoost':'#2E6DA4','LightGBM':'#E05C5C','LSTM':'#10B981','ARIMA':'#F59E0B','SARIMA':'#6366F1'}
RMSE_INFO   = {'XGBoost':'~0.49°C','LightGBM':'~0.44°C','LSTM':'~1.99°C','ARIMA':'~1.74°C','SARIMA':'~1.73°C','Ensemble':'~0.97°C'}
R2_INFO     = {'XGBoost':'~0.994','LightGBM':'~0.995','LSTM':'~0.901','ARIMA':'~0.924','SARIMA':'~0.926','Ensemble':'~0.976'}
TYPE_INFO   = {'XGBoost':'Gradient Boosting','LightGBM':'Gradient Boosting',
               'LSTM':'Deep Learning','ARIMA':'Statistical','SARIMA':'Statistical (Seasonal)',
               'Ensemble':'Weighted Average'}
MODEL_ORDER = ['XGBoost','LightGBM','LSTM','ARIMA','SARIMA']


# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌤️ Delhi Weather Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-model forecast &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; ARIMA &nbsp;·&nbsp; SARIMA &nbsp;·&nbsp; Ensemble &nbsp;|&nbsp; Trained on Delhi 2013–2017</div>', unsafe_allow_html=True)

if not core_ok:
    st.error("⚠️ Core models not found. Run the training notebook in Colab first.")
    st.stop()

# ── Model Status Row ───────────────────────────────────────────────────────────
lstm_fully_ready = models['lstm'] is not None and models['lstm_scaler'] is not None
model_status = [
    ('XGBoost',  models['xgb']    is not None),
    ('LightGBM', models['lgb']    is not None),
    ('LSTM',     lstm_fully_ready),
    ('ARIMA',    models['arima']  is not None),
    ('SARIMA',   models['sarima'] is not None),
    ('Ensemble', True),
]
for col, (name, loaded) in zip(st.columns(6), model_status):
    badge = 'badge-active' if loaded else 'badge-missing'
    label = '✓ loaded' if loaded else '○ not trained'
    col.markdown(
        f'<div style="text-align:center">'
        f'<div style="font-size:0.8rem;font-weight:600;color:#1E3A5F">{name}</div>'
        f'<span class="{badge}">{label}</span></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Today's Weather")
    st.markdown("*Enter observed values for today:*")
    st.markdown("---")

    temp       = st.slider("🌡️ Mean Temperature (°C)", 0.0, 45.0, 22.0, 0.5)
    humidity   = st.slider("💧 Humidity (%)",          10.0,100.0, 65.0, 1.0)
    wind_speed = st.slider("💨 Wind Speed (km/h)",     0.0, 50.0,  8.0, 0.5)
    pressure   = st.slider("🌀 Mean Pressure (hPa)",  990.0,1030.0,1010.0, 0.5)

    st.markdown("---")
    st.markdown("### 📅 Recent History")
    st.caption("Used for lag features and ARIMA/SARIMA context")
    temp_yesterday = st.number_input("Yesterday's Temp (°C)", value=21.0, step=0.5)
    temp_2days_ago = st.number_input("2 Days Ago Temp (°C)",  value=20.0, step=0.5)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Tomorrow's Temperature")

    # Show LSTM status / error
    if models['lstm'] is not None and models['lstm_scaler'] is None:
        st.error(
            "⚠️ **LSTM scaler missing**\n\n"
            "`lstm_scaler.pkl` not in `models/`.\n\n"
            "In Colab Step 8, after fitting your scaler add:\n"
            "```python\n"
            "joblib.dump(scaler, 'models/lstm_scaler.pkl')\n"
            "```\n"
            "then re-push (Step 11)."
        )
    elif st.session_state.get('_lstm_runtime_error'):
        st.error(f"⚠️ LSTM runtime error:\n```\n{st.session_state['_lstm_runtime_error']}\n```")

    st.markdown("""
    <div class="note-box">
    💡 LSTM requires <code>lstm_model.keras</code> <b>and</b>
    <code>lstm_scaler.pkl</code> in <code>models/</code>.
    Run all Colab steps and re-push.
    </div>""", unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    if predict_btn:
        st.session_state['_lstm_runtime_error'] = ''
        with st.spinner("Running all models..."):
            results = run_predictions(temp, humidity, wind_speed, pressure,
                                      temp_yesterday, temp_2days_ago)

        ens = results.get('Ensemble')
        if ens:
            st.markdown(f"""
            <div class="pred-card ensemble" style="padding:1.5rem;margin-bottom:1rem">
                <div class="pred-model-name">🏆 Ensemble Forecast</div>
                <div class="pred-value">{ens:.1f}°C</div>
                <div class="pred-condition">{condition(ens)}</div>
                <div style="font-size:0.75rem;opacity:0.7;margin-top:0.3rem">
                    Weighted average of {len(results)-1} model(s)
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Individual Model Predictions</div>',
                    unsafe_allow_html=True)
        for col, name in zip(st.columns(5), MODEL_ORDER):
            pred = results.get(name)
            cls  = CARD_CLASS.get(name, '')
            with col:
                if pred is not None:
                    st.markdown(f"""
                    <div class="pred-card {cls}">
                        <div class="pred-model-name">{name}</div>
                        <div class="pred-value">{pred:.1f}°C</div>
                        <div class="pred-condition">{condition(pred)}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-card" style="opacity:0.4">
                        <div class="pred-model-name">{name}</div>
                        <div class="pred-na">N/A</div>
                        <div class="pred-condition" style="font-size:0.72rem">not loaded</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Model Comparison Chart</div>',
                    unsafe_allow_html=True)
        chart_keys = [k for k in MODEL_ORDER if k in results]
        chart_vals = [results[k] for k in chart_keys]

        if chart_keys:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            fig.patch.set_facecolor('#F8FAFC')
            ax.set_facecolor('#F8FAFC')
            colors = [COLOR_MAP.get(k,'#888') for k in chart_keys]
            bars   = ax.bar(chart_keys, chart_vals, color=colors, edgecolor='white', width=0.55, zorder=3)
            if ens:
                ax.axhline(ens, color='#7C3AED', lw=1.8, ls='--',
                           alpha=0.8, label=f'Ensemble: {ens:.1f}°C', zorder=4)
                ax.legend(fontsize=9, framealpha=0.7)
            y_pad = max(chart_vals) - min(chart_vals) if len(chart_vals) > 1 else 1
            ax.set_ylim(min(chart_vals) - max(2.5, y_pad*0.3),
                        max(chart_vals) + max(3.0, y_pad*0.4))
            ax.set_ylabel('Predicted Temperature (°C)', fontsize=10)
            ax.set_title('All Model Predictions', fontsize=11, fontweight='bold', color='#0F2942', pad=10)
            for bar, val in zip(bars, chart_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                        f'{val:.1f}°C', ha='center', va='bottom', fontsize=9, fontweight='600', color='#0F2942')
            ax.spines[['top','right']].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')
            ax.tick_params(colors='#475569')
            ax.grid(axis='y', color='#E2E8F0', lw=0.8, zorder=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-title">Summary Table</div>', unsafe_allow_html=True)
        rows = []
        for name in MODEL_ORDER:
            pred = results.get(name)
            rows.append({'Model': name, 'Type': TYPE_INFO[name],
                         'Prediction': f"{pred:.1f}°C" if pred is not None else 'N/A',
                         'Condition':  condition(pred) if pred is not None else '—',
                         'RMSE': RMSE_INFO[name], 'R²': R2_INFO[name]})
        if ens:
            rows.append({'Model':'🏆 Ensemble','Type':'Weighted Average',
                         'Prediction':f"{ens:.1f}°C",'Condition':condition(ens),
                         'RMSE':'~0.97°C','R²':'~0.976'})
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    else:
        loaded_names  = [n for n,l in model_status if l and n != 'Ensemble']
        missing_names = [n for n,l in model_status if not l and n != 'Ensemble']
        st.markdown("""
        <div style="background:#F0F7FF;border-radius:14px;padding:2rem;
                    text-align:center;margin-top:1rem;border:1px solid #BFDBFE">
            <div style="font-size:3rem">🌤️</div>
            <div style="font-size:1.2rem;font-weight:600;color:#1E3A5F;margin:0.5rem 0">Ready to Forecast</div>
            <div style="color:#5B7FA6;font-size:0.95rem">
                Enter today's weather values in the sidebar<br>
                and press <b>Predict Tomorrow's Temperature</b>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Model Status</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Loaded ({len(loaded_names)}):** " + " · ".join(loaded_names))
        with c2:
            if missing_names:
                st.warning(f"**Not trained ({len(missing_names)}):** " + " · ".join(missing_names))
            else:
                st.success("All models loaded! ✓")


# ── Right Panel ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Feature Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box"><b>🌡️ Temperature</b><br>Mean daily temp in °C. Delhi: ~6°C (Jan) to ~40°C (May–Jun).</div>
    <div class="info-box"><b>💧 Humidity</b><br>Relative humidity %. Peaks in monsoon (Jul–Sep).</div>
    <div class="info-box"><b>💨 Wind Speed</b><br>Mean daily wind in km/h. Strong winds often precede storms.</div>
    <div class="info-box"><b>🌀 Pressure</b><br>Atmospheric pressure in hPa. Low = rain likely. Normal = 1005–1020.</div>
    <div class="info-box"><b>📅 Recent History</b><br>Yesterday + 2-days-ago temp feed lag features for XGB/LGB and context window for ARIMA/SARIMA.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Model Reference</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Model'  : ['XGBoost','LightGBM','LSTM','ARIMA','SARIMA','Ensemble'],
        'RMSE'   : ['~0.49°C','~0.44°C','~1.99°C','~1.74°C','~1.73°C','~0.97°C'],
        'R²'     : ['~0.994','~0.995','~0.901','~0.924','~0.926','~0.976'],
    }).set_index('Model'), use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Ensemble Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Uses <b>inverse-RMSE weighting</b> with actual training RMSE values.
    LightGBM (0.44) and XGBoost (0.49) carry ~80% of ensemble weight.
    ARIMA/SARIMA/LSTM have much lower weight due to higher RMSE.
    Weights auto-renormalise if any model is missing.
    </div>""", unsafe_allow_html=True)

    if predict_btn and 'results' in dir():
        st.markdown('<div class="section-title" style="margin-top:1.2rem">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Parameter': ['Temperature','Humidity','Wind Speed','Pressure','Temp Yesterday','Temp 2 Days Ago'],
            'Value'    : [f'{temp}°C', f'{humidity}%', f'{wind_speed} km/h',
                          f'{pressure} hPa', f'{temp_yesterday}°C', f'{temp_2days_ago}°C'],
        }).set_index('Parameter'), use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🌦️ Delhi Weather Predictor · Dataset: Delhi 2013–2017 · "
           "Models: XGBoost · LightGBM · LSTM · ARIMA · SARIMA · Ensemble · Built with Streamlit")
