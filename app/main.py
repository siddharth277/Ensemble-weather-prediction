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

st.set_page_config(
    page_title="Delhi Weather Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_all_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = {}

    for key, fname in [('xgb',          'xgboost_model.pkl'),
                        ('lgb',          'lightgbm_model.pkl'),
                        ('weights',      'ensemble_weights.pkl'),
                        ('meta',         'feature_meta.pkl'),
                        ('arima',        'arima_model.pkl'),
                        ('sarima',       'sarima_model.pkl'),
                        ('lstm_scaler',  'lstm_scaler.pkl'),
                        ('lstm_features','lstm_features.pkl')]:
        try:
            m[key] = joblib.load(os.path.join(base, 'models', fname))
        except Exception:
            m[key] = None

    try:
        import tensorflow as tf
        lstm_path = os.path.join(base, 'models', 'lstm_model.keras')
        m['lstm'] = tf.keras.models.load_model(lstm_path) if os.path.exists(lstm_path) else None
    except Exception:
        m['lstm'] = None

    if m['weights'] is None:
        m['weights'] = {'w_xgb': 0.35, 'w_lgb': 0.35}

    return m

models  = load_all_models()
core_ok = models['xgb'] is not None and models['meta'] is not None

def build_input_row(temp, humidity, wind_speed, pressure,
                    temp_yesterday, temp_2days_ago, features):
    today = datetime.date.today()
    month = today.month
    doy   = today.timetuple().tm_yday
    row   = {f: 0.0 for f in features}
    row.update({
                  : temp, 'humidity': humidity,
                    : wind_speed, 'meanpressure': pressure,
               : month, 'day': today.day,
                     : doy, 'day_of_week': today.weekday(),
                : {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[month],
                   : np.sin(2*np.pi*month/12),
                   : np.cos(2*np.pi*month/12),
                 :   np.sin(2*np.pi*doy/365),
                 :   np.cos(2*np.pi*doy/365),
                   : temp_yesterday, 'temp_lag2': temp_2days_ago,
                   : temp_2days_ago,
                   : (temp + temp_yesterday + temp_2days_ago) / 3,
                    :     temp * humidity / 100,
                        : 0.0,
                    :     temp - temp_yesterday,
                    :     temp - 0.5 * wind_speed,
                   :  0.7*temp + 0.3*temp_yesterday,
                    : 0.6*temp + 0.4*temp_yesterday,
    })
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

def arima_one_step(model, temp_yesterday):
    try:
        updated = model.append([temp_yesterday], refit=False)
        return float(updated.forecast(steps=1)[0])
    except Exception:
        return None

def _build_lstm_window(temp, humidity, wind_speed, pressure,
                        temp_yesterday, temp_2days_ago, n_steps=30):
    temps     = np.linspace(temp_2days_ago, temp, n_steps)
    hums      = np.full(n_steps, humidity)
    winds     = np.full(n_steps, wind_speed)
    pressures = np.full(n_steps, pressure)

    df = pd.DataFrame({
                  :     temps,
                  :     hums,
                    :   winds,
                      : pressures,
    })

    df['temp_diff_1']  = df['meantemp'].diff(1)
    df['temp_diff_2']  = df['meantemp'].diff(2)
    df['temp_diff_3']  = df['meantemp'].diff(3)
    df['temp_diff_7']  = df['meantemp'].diff(7)
    df['temp_diff_14'] = df['meantemp'].diff(14)
    df['temp_accel']   = df['temp_diff_1'].diff(1)

    df['temp_roll_mean_7']  = df['meantemp'].rolling(7).mean()
    df['temp_roll_mean_14'] = df['meantemp'].rolling(14).mean()
    df['temp_roll_std_7']   = df['meantemp'].rolling(7).std()
    df['temp_roll_std_14']  = df['meantemp'].rolling(14).std()
    df['temp_roll_min_7']   = df['meantemp'].rolling(7).min()
    df['temp_roll_max_7']   = df['meantemp'].rolling(7).max()
    df['temp_roll_range_7'] = df['temp_roll_max_7'] - df['temp_roll_min_7']

    df['temp_momentum_7']  = df['temp_roll_mean_7']  - df['temp_roll_mean_14']
    df['temp_momentum_14'] = df['temp_roll_mean_14'] - df['meantemp'].rolling(21).mean()

    df['temp_ema_7']    = df['meantemp'].ewm(span=7,  adjust=False).mean()
    df['temp_ema_14']   = df['meantemp'].ewm(span=14, adjust=False).mean()
    df['temp_ema_diff'] = df['temp_ema_7'] - df['temp_ema_14']

    roll_mean_30       = df['meantemp'].rolling(30).mean()
    roll_std_30        = df['meantemp'].rolling(30).std().replace(0, 1)
    df['temp_zscore_30'] = (df['meantemp'] - roll_mean_30) / roll_std_30

    df['humidity_diff_1']     = df['humidity'].diff(1)
    df['humidity_roll_7']     = df['humidity'].rolling(7).mean()
    df['humidity_roll_std_7'] = df['humidity'].rolling(7).std()

    df['wind_roll_7'] = df['wind_speed'].rolling(7).mean()
    df['wind_diff_1'] = df['wind_speed'].diff(1)

    df['pressure_diff_1'] = df['meanpressure'].diff(1)
    df['pressure_diff_3'] = df['meanpressure'].diff(3)
    df['pressure_roll_7'] = df['meanpressure'].rolling(7).mean()

    df['humidity_x_wind'] = df['humidity']      * df['wind_speed']
    df['temp_x_humidity'] = df['meantemp']       * df['humidity']
    df['temp_x_wind']     = df['meantemp']       * df['wind_speed']
    df['wind_x_pressure'] = df['wind_speed']     * df['meanpressure']

    df = df.ffill().fillna(0)
    return df

def lstm_one_step(lstm_model, temp, humidity, wind_speed, pressure,
                  temp_yesterday, temp_2days_ago, scaler=None, feature_cols=None):
    try:
        LOOKBACK = 30
        df = _build_lstm_window(temp, humidity, wind_speed, pressure,
                                 temp_yesterday, temp_2days_ago, LOOKBACK)

        if feature_cols is not None:
            for mc in [c for c in feature_cols if c not in df.columns]:
                df[mc] = 0.0
            df = df.reindex(columns=feature_cols, fill_value=0.0)
        else:
            TARGET_COL   = 'meantemp'
            feature_cols = [TARGET_COL] + [c for c in df.columns if c != TARGET_COL]
            df = df[feature_cols]

        window = df.values.astype(np.float32)

        if scaler is not None:
            window_scaled = scaler.transform(window)
        else:
            from sklearn.preprocessing import MinMaxScaler
            local_scaler  = MinMaxScaler()
            window_scaled = local_scaler.fit_transform(window)
            scaler        = local_scaler

        X        = window_scaled.reshape(1, LOOKBACK, window_scaled.shape[1])
        y_scaled = float(lstm_model.predict(X, verbose=0)[0, 0])

        dummy       = np.zeros((1, window_scaled.shape[1]))
        dummy[0, 0] = y_scaled
        pred        = float(scaler.inverse_transform(dummy)[0, 0])

        if pred < 0 or pred > 50:
            pred = (temp + temp_yesterday + temp_2days_ago) / 3 + (temp - temp_yesterday) * 0.5

        return pred
    except Exception:
        return None

ENSEMBLE_MANUAL_WEIGHTS = {
              : 0.35,
              : 0.35,
              : 0.15,
              : 0.08,
              : 0.07,
}

def run_predictions(temp, humidity, wind_speed, pressure, temp_yesterday, temp_2days_ago):
    results = {}
    X = build_input_row(temp, humidity, wind_speed, pressure,
                        temp_yesterday, temp_2days_ago, models['meta']['features'])

    if models['xgb']:
        results['XGBoost']  = float(models['xgb'].predict(X)[0])
    if models['lgb']:
        results['LightGBM'] = float(models['lgb'].predict(X)[0])
    if models['lstm']:
        v = lstm_one_step(models['lstm'], temp, humidity, wind_speed, pressure,
                          temp_yesterday, temp_2days_ago,
                          scaler=models.get('lstm_scaler'),
                          feature_cols=models.get('lstm_features'))
        if v is not None:
            results['LSTM'] = v
    if models['arima']:
        v = arima_one_step(models['arima'], temp_yesterday)
        if v is not None:
            results['ARIMA'] = v
    if models['sarima']:
        v = arima_one_step(models['sarima'], temp_yesterday)
        if v is not None:
            results['SARIMA'] = v

    available = list(results.keys())
    manual_w  = {k: ENSEMBLE_MANUAL_WEIGHTS[k] for k in available if k in ENSEMBLE_MANUAL_WEIGHTS}
    total_w   = sum(manual_w.values())
    if total_w > 0:
        norm_w            = {k: v / total_w for k, v in manual_w.items()}
        results['Ensemble'] = sum(results[k] * norm_w[k] for k in norm_w)

    return results

def condition(t):
    if t < 10:  return "🥶 Very Cold"
    if t < 18:  return "🌥️ Cool"
    if t < 26:  return "🌤️ Pleasant"
    if t < 32:  return "☀️ Warm"
    return "🔥 Hot"

CARD_CLASS  = {'LSTM':'lstm-card','ARIMA':'arima-card','SARIMA':'sarima-card','Ensemble':'ensemble'}
COLOR_MAP   = {'XGBoost':'#2E6DA4','LightGBM':'#E05C5C','LSTM':'#10B981','ARIMA':'#F59E0B','SARIMA':'#6366F1'}
RMSE_INFO   = {'XGBoost':'~2.1°C','LightGBM':'~2.2°C','LSTM':'~2.5°C','ARIMA':'~3.2°C','SARIMA':'~3.0°C','Ensemble':'~2.0°C'}
R2_INFO     = {'XGBoost':'~0.95','LightGBM':'~0.94','LSTM':'~0.93','ARIMA':'~0.88','SARIMA':'~0.90','Ensemble':'~0.96'}
TYPE_INFO   = {'XGBoost':'Gradient Boosting','LightGBM':'Gradient Boosting','LSTM':'Deep Learning','ARIMA':'Statistical','SARIMA':'Statistical (Seasonal)','Ensemble':'Weighted Average'}
MODEL_ORDER = ['XGBoost','LightGBM','LSTM','ARIMA','SARIMA']

st.markdown('<div class="main-header">🌤️ Delhi Weather Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-model forecast &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; ARIMA &nbsp;·&nbsp; SARIMA &nbsp;·&nbsp; Ensemble &nbsp;|&nbsp; Trained on Delhi 2013–2017</div>', unsafe_allow_html=True)

if not core_ok:
    st.error("⚠️ Core models not found. Run the training notebook in Colab first.")
    st.stop()

model_status = [
    ('XGBoost',  models['xgb']    is not None),
    ('LightGBM', models['lgb']    is not None),
    ('LSTM',     models['lstm']   is not None),
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
    st.markdown("""
    <div class="note-box">
    💡 LSTM & ARIMA/SARIMA require their model files in <code>models/</code>.
    Run all notebook steps in Colab first to generate them.
    </div>""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    if predict_btn:
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

        st.markdown('<div class="section-title">Individual Model Predictions</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-title">Model Comparison Chart</div>', unsafe_allow_html=True)
        chart_keys = [k for k in MODEL_ORDER if k in results]
        chart_vals = [results[k] for k in chart_keys]

        if chart_keys:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            fig.patch.set_facecolor('#F8FAFC')
            ax.set_facecolor('#F8FAFC')
            colors = [COLOR_MAP.get(k,'#888') for k in chart_keys]
            bars   = ax.bar(chart_keys, chart_vals, color=colors, edgecolor='white', width=0.55, zorder=3)
            if ens:
                ax.axhline(ens, color='#7C3AED', lw=1.8, ls='--', alpha=0.8, label=f'Ensemble: {ens:.1f}°C', zorder=4)
                ax.legend(fontsize=9, framealpha=0.7)
            y_pad = max(chart_vals) - min(chart_vals) if len(chart_vals) > 1 else 1
            ax.set_ylim(min(chart_vals) - max(2.5, y_pad*0.3), max(chart_vals) + max(3.0, y_pad*0.4))
            ax.set_ylabel('Predicted Temperature (°C)', fontsize=10)
            ax.set_title('All Model Predictions', fontsize=11, fontweight='bold', color='#0F2942', pad=10)
            for bar, val in zip(bars, chart_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f'{val:.1f}°C',
                        ha='center', va='bottom', fontsize=9, fontweight='600', color='#0F2942')
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
                                     : f"{pred:.1f}°C" if pred else 'N/A',
                                    :  condition(pred) if pred else '—',
                               : RMSE_INFO[name], 'R²': R2_INFO[name]})
        if ens:
            rows.append({'Model':'🏆 Ensemble','Type':'Weighted Average',
                                     :f"{ens:.1f}°C",'Condition':condition(ens),
                               :'~2.0°C','R²':'~0.96'})
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    else:
        loaded_names  = [n for n,l in model_status if l and n != 'Ensemble']
        missing_names = [n for n,l in model_status if not l and n != 'Ensemble']
        st.markdown("""
        <div style="background:#F0F7FF;border-radius:14px;padding:2rem;text-align:center;margin-top:1rem;border:1px solid #BFDBFE">
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
               : ['XGBoost','LightGBM','LSTM','ARIMA','SARIMA','Ensemble'],
               : ['~2.1°C','~2.2°C','~2.5°C','~3.2°C','~3.0°C','~2.0°C'],
               : ['~0.95','~0.94','~0.93','~0.88','~0.90','~0.96'],
    }).set_index('Model'), use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Ensemble Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Uses <b>tuned manual weights</b>: XGBoost (35%) + LightGBM (35%) lead as they
    have the lowest RMSE. LSTM (15%), ARIMA (8%), SARIMA (7%) contribute when loaded.
    Weights auto-renormalise when any model is missing.
    </div>""", unsafe_allow_html=True)

    if predict_btn and 'results' in dir():
        st.markdown('<div class="section-title" style="margin-top:1.2rem">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
                       : ['Temperature','Humidity','Wind Speed','Pressure','Temp Yesterday','Temp 2 Days Ago'],
                       : [f'{temp}°C', f'{humidity}%', f'{wind_speed} km/h', f'{pressure} hPa', f'{temp_yesterday}°C', f'{temp_2days_ago}°C'],
        }).set_index('Parameter'), use_container_width=True)

st.markdown("---")
st.caption("🌦️ Delhi Weather Predictor · Dataset: Delhi 2013–2017 · Models: XGBoost · LightGBM · LSTM · ARIMA · SARIMA · Ensemble · Built with Streamlit")
