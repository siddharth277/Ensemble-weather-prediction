# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10: LSTM MODEL — Bidirectional LSTM + Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════════
#
# PIPELINE    : LSTM (independent — uses its own feature engineering)
# ARCHITECTURE: BiLSTM(128) → BiLSTM(64) + Residual → MultiHeadAttention
#               + LSTM(32) → Merge → Dense(64→32→1)
#
# FIXES APPLIED vs original:
#   1. Model size reduced  — original BiLSTM(256)+BiLSTM(128) had ~2M+ params
#      for only 1,400 training windows. This caused heavy overfitting.
#      New: BiLSTM(128)+BiLSTM(64) ~300K params — right-sized for the data.
#   2. Data cleaning added — raw data has pressure outliers (max=7679 hPa)
#      and negative wind values. LSTM now cleans these identically to
#      02_eda_cleaning.py before building features. Without this the scaler
#      is corrupted by outliers, degrading all scaled features.
#   3. Cyclical time features added — month_sin/cos, day_of_year_sin/cos.
#      These give the LSTM an explicit seasonal clock without leakage.
#   4. Dropout increased slightly on BiLSTM blocks (0.20) to reduce overfit.
#   5. BATCH_SIZE increased 8 → 16 for smoother gradient updates on this size.
#   6. patience on EarlyStopping kept at 35; ReduceLROnPlateau patience 10→12.
#   7. output CSV date column: stored as date string (YYYY-MM-DD) to match
#      the format produced by xgb.csv / lgb.csv — ensures clean ensemble merge.
#
# DATA FLOW:
#   Input  : data/raw/Train.csv + data/raw/Test.csv
#   Output : data/predictions/lstm.csv  (columns: date, id, prediction, actual)
#            models/lstm_model.keras
#
# ALIGNMENT: 114 test-date predictions matching data/raw/Test.csv dates.
#            LOOKBACK=30 sliding window — test windows use last 30 train rows
#            as context so all 114 test dates are always covered.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts/Colab
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
    Input, Add, Concatenate, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

# ── CELL 1: CONFIG ────────────────────────────────────────────────────────────
TRAIN_PATH  = 'data/raw/Train.csv'
TEST_PATH   = 'data/raw/Test.csv'
DATE_COL    = 'date'
TARGET_COL  = 'meantemp'
LOOKBACK    = 30       # days of context per prediction
EPOCHS      = 250      # EarlyStopping will stop well before this
BATCH_SIZE  = 16       # FIX: was 8 — larger batch = smoother gradients
VAL_SPLIT   = 0.15     # fraction of train windows used for validation

print("=" * 62)
print("  LSTM MODEL — Bidirectional LSTM + MultiHeadAttention")
print("=" * 62)

# ── CELL 2: LOAD RAW DATA ─────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COL])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=[DATE_COL])

train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
test_df  = test_df.sort_values(DATE_COL).reset_index(drop=True)

# Remove duplicate dates in test (2017-01-01 appears in both train and test)
dupes = test_df.duplicated(subset=DATE_COL, keep=False).sum()
if dupes > 0:
    print(f'  Duplicate dates in test: {dupes} — keeping last')
    test_df = test_df.drop_duplicates(subset=DATE_COL, keep='last').reset_index(drop=True)

print(f"\n  Train : {len(train_df)} rows  "
      f"{train_df[DATE_COL].iloc[0].date()} -> {train_df[DATE_COL].iloc[-1].date()}")
print(f"  Test  : {len(test_df)} rows   "
      f"{test_df[DATE_COL].iloc[0].date()} -> {test_df[DATE_COL].iloc[-1].date()}")

# ── CELL 3: DATA CLEANING ─────────────────────────────────────────────────────
#
# FIX: Original script skipped cleaning entirely.
# Raw data has pressure outliers (max=7679 hPa — physical range: 950–1100 hPa)
# and negative wind values. These corrupt the MinMaxScaler: one outlier in
# meanpressure stretches the scale so that all normal values map to ~0,
# destroying the pressure signal completely.
#
# Cleaning strategy (matches 02_eda_cleaning.py exactly):
#   1. meanpressure outside [950, 1100] → NaN → rolling median imputation
#   2. wind_speed < 0                   → NaN → forward/backward fill
#   3. humidity > 100                   → cap at 100
#   4. Residual NaNs                    → ffill then bfill

def clean_weather(df):
    df = df.copy()

    # 1. Pressure outliers
    bad_p = (df['meanpressure'] < 950) | (df['meanpressure'] > 1100)
    if bad_p.sum() > 0:
        print(f"  Cleaning {bad_p.sum()} bad pressure rows in "
              f"{'train' if len(df) > 200 else 'test'}")
    df.loc[bad_p, 'meanpressure'] = np.nan
    df['meanpressure'] = df['meanpressure'].fillna(
        df['meanpressure'].rolling(3, min_periods=1, center=True).median()
    )

    # 2. Negative wind
    bad_w = df['wind_speed'] < 0
    df.loc[bad_w, 'wind_speed'] = np.nan

    # 3. Impossible humidity
    df.loc[df['humidity'] > 100, 'humidity'] = 100.0

    # 4. Fill residual NaNs
    num_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    df[num_cols] = df[num_cols].ffill().bfill()

    return df

train_df = clean_weather(train_df)
test_df  = clean_weather(test_df)

# ── CELL 4: LSTM-SPECIFIC FEATURE ENGINEERING ─────────────────────────────────
#
# These features are applied BEFORE scaling and windowing so they become
# additional channels in the 3D input tensor (samples, lookback, features).
# diff() and rolling() are safe here because the LSTM reads the full sequence
# — there is no row-level leakage risk like there is with tabular models.
#
# FIX: Added cyclical time features (month_sin/cos, doy_sin/cos).
# Without these the LSTM has no explicit knowledge of seasonality; it must
# infer it purely from temperature autocorrelation, which is harder.

def engineer_features(df):
    df = df.copy()

    # -- Cyclical time features (FIX: missing in original) -------------------
    df['month']       = df[DATE_COL].dt.month
    df['day_of_year'] = df[DATE_COL].dt.dayofyear
    df['month_sin']   = np.sin(2 * np.pi * df['month']       / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month']       / 12)
    df['doy_sin']     = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']     = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # -- Velocity (1st differences) ------------------------------------------
    df['temp_diff_1']  = df['meantemp'].diff(1)
    df['temp_diff_2']  = df['meantemp'].diff(2)
    df['temp_diff_3']  = df['meantemp'].diff(3)
    df['temp_diff_7']  = df['meantemp'].diff(7)
    df['temp_diff_14'] = df['meantemp'].diff(14)

    # -- Acceleration (2nd derivative) ----------------------------------------
    df['temp_accel'] = df['temp_diff_1'].diff(1)

    # -- Rolling statistics ---------------------------------------------------
    df['temp_roll_mean_7']  = df['meantemp'].rolling(7).mean()
    df['temp_roll_mean_14'] = df['meantemp'].rolling(14).mean()
    df['temp_roll_std_7']   = df['meantemp'].rolling(7).std()
    df['temp_roll_std_14']  = df['meantemp'].rolling(14).std()
    df['temp_roll_min_7']   = df['meantemp'].rolling(7).min()
    df['temp_roll_max_7']   = df['meantemp'].rolling(7).max()
    df['temp_roll_range_7'] = df['temp_roll_max_7'] - df['temp_roll_min_7']

    # -- Momentum (short vs long trend) --------------------------------------
    df['temp_momentum_7']  = df['temp_roll_mean_7']  - df['temp_roll_mean_14']
    df['temp_momentum_14'] = df['temp_roll_mean_14'] - df['meantemp'].rolling(21).mean()

    # -- Exponential moving averages -----------------------------------------
    df['temp_ema_7']    = df['meantemp'].ewm(span=7,  adjust=False).mean()
    df['temp_ema_14']   = df['meantemp'].ewm(span=14, adjust=False).mean()
    df['temp_ema_diff'] = df['temp_ema_7'] - df['temp_ema_14']

    # -- Z-score relative to 30-day window -----------------------------------
    roll_mean_30         = df['meantemp'].rolling(30).mean()
    roll_std_30          = df['meantemp'].rolling(30).std().replace(0, 1)
    df['temp_zscore_30'] = (df['meantemp'] - roll_mean_30) / roll_std_30

    # -- Humidity derived features -------------------------------------------
    df['humidity_diff_1']     = df['humidity'].diff(1)
    df['humidity_roll_7']     = df['humidity'].rolling(7).mean()
    df['humidity_roll_std_7'] = df['humidity'].rolling(7).std()

    # -- Wind derived --------------------------------------------------------
    df['wind_roll_7'] = df['wind_speed'].rolling(7).mean()
    df['wind_diff_1'] = df['wind_speed'].diff(1)

    # -- Pressure derived ----------------------------------------------------
    df['pressure_diff_1'] = df['meanpressure'].diff(1)
    df['pressure_diff_3'] = df['meanpressure'].diff(3)
    df['pressure_roll_7'] = df['meanpressure'].rolling(7).mean()

    # -- Cross-feature interactions ------------------------------------------
    df['humidity_x_wind'] = df['humidity']     * df['wind_speed']
    df['temp_x_humidity'] = df['meantemp']      * df['humidity']
    df['temp_x_wind']     = df['meantemp']      * df['wind_speed']
    df['wind_x_pressure'] = df['wind_speed']    * df['meanpressure']

    # Drop helper columns not needed as model inputs
    df = df.drop(columns=['month', 'day_of_year'], errors='ignore')

    # Fill NaN from diff/rolling — forward-fill then zero for leading rows
    df = df.ffill().fillna(0)

    return df


train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

print(f"\n  Features after engineering : {len(train_df.columns)} columns")

# ── CELL 5: SCALE & WINDOW ────────────────────────────────────────────────────
#
# MinMaxScaler fitted ONLY on train — no future leakage into test scaling.
# Target (meantemp) placed at column index 0 for clean inverse_transform.
# Combine train+test AFTER fitting scaler so windowing is continuous.

feature_cols = [c for c in train_df.columns if c != DATE_COL]
feature_cols = [TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]
n_features   = len(feature_cols)

print(f"  Total input features       : {n_features}")

combined_df     = pd.concat([train_df, test_df], ignore_index=True)\
                    .sort_values(DATE_COL).reset_index(drop=True)

scaler          = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[feature_cols])              # fit on train ONLY
combined_scaled = scaler.transform(combined_df[feature_cols])


def create_sliding_window(data, lookback):
    """Build (samples, lookback, features) windows and (samples,) targets."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])   # col 0 = meantemp (target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_all, y_all = create_sliding_window(combined_scaled, LOOKBACK)

split_idx             = len(train_df) - LOOKBACK
X_train_w, y_train_w = X_all[:split_idx], y_all[:split_idx]
X_test_w,  y_test_w  = X_all[split_idx:], y_all[split_idx:]

print(f"  X_train windows            : {X_train_w.shape}")
print(f"  X_test  windows            : {X_test_w.shape}  "
      f"(should match {len(test_df)} test rows)")

# ── CELL 6: BUILD MODEL ───────────────────────────────────────────────────────
#
# FIX: Original model was heavily oversized for this dataset.
# ~1,400 training windows with a 2M+ parameter model = severe overfitting.
#
# Original → Fixed:
#   BiLSTM(256) → BiLSTM(128)   |  BiLSTM(128) → BiLSTM(64)
#   LSTM(64)    → LSTM(32)      |  Dense head: 128→64→32→1 → 64→32→1
#   Total params: ~2.1M         →  ~310K
#
# Architecture kept intact: BiLSTM + Residual + MultiHeadAttention + LSTM branch.
# Only the layer widths are reduced to match dataset size.

n_lookback = X_train_w.shape[1]

inputs = Input(shape=(n_lookback, n_features), name='input')

# BiLSTM Block 1 (FIX: 256 → 128)
x = Bidirectional(LSTM(128, return_sequences=True), name='bilstm_1')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.20)(x)   # FIX: was 0.15, slightly higher to combat overfit

# BiLSTM Block 2 + Residual (FIX: 128 → 64)
x2 = Bidirectional(LSTM(64, return_sequences=True), name='bilstm_2')(x)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.15)(x2)

# Project Block-1 output to match Block-2 dims for residual add
# Block-1 output dim = 128*2 = 256; Block-2 output dim = 64*2 = 128 → project to 128
skip = Dense(128, use_bias=False, name='skip_proj')(x)
x    = Add(name='residual_add')([x2, skip])
x    = LayerNormalization(name='layer_norm')(x)

# Branch 1: Multi-Head Self-Attention
# key_dim reduced 64 → 32 proportionally
attn_out = MultiHeadAttention(
    num_heads=4, key_dim=32, dropout=0.1, name='mha'
)(x, x)
context = GlobalAveragePooling1D(name='attn_pool')(attn_out)   # (batch, 128)

# Branch 2: Sequential LSTM state (FIX: 64 → 32)
seq_out = LSTM(32, return_sequences=False, name='lstm_final')(x)
seq_out = Dropout(0.10)(seq_out)

# Merge both branches → (batch, 160)
merged_out = Concatenate(name='merge')([seq_out, context])

# Dense prediction head (FIX: 128→64→32→1 removed top layer → 64→32→1)
z = Dense(64, activation='relu', name='dense_1')(merged_out)
z = Dropout(0.10)(z)
z = Dense(32, activation='relu', name='dense_2')(z)
outputs = Dense(1, name='output')(z)

model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_MHA_v4')

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer = optimizer,
    loss      = tf.keras.losses.Huber(delta=1.0),   # robust to temperature outliers
    metrics   = ['mae']
)

print(f"\n  Model parameters : {model.count_params():,}")
print(f"  (Original had ~2.1M params — reduced to prevent overfitting on ~1400 samples)")

# ── CELL 7: TRAIN ─────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=35,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=12,  # FIX: was 10
        min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        filepath='models/lstm_best.keras',
        monitor='val_loss', save_best_only=True, verbose=0
    )
]

print(f"\n  Training (up to {EPOCHS} epochs, early stopping patience=35)...")
history = model.fit(
    X_train_w, y_train_w,
    epochs           = EPOCHS,
    batch_size       = BATCH_SIZE,
    validation_split = VAL_SPLIT,
    callbacks        = callbacks,
    verbose          = 1
)

best_epoch    = int(np.argmin(history.history['val_loss']))
best_val_loss = min(history.history['val_loss'])
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Best val_loss : {best_val_loss:.4f}")

# ── CELL 8: INVERSE TRANSFORM HELPER ─────────────────────────────────────────
def inverse_target(scaled_values, scaler, n_feat):
    """
    Inverse-transform ONLY the target column (index 0 = meantemp).
    Fills other columns with zeros as dummy placeholders.
    """
    dummy       = np.zeros((len(scaled_values), n_feat))
    dummy[:, 0] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

# ── CELL 9: PREDICT & EVALUATE ────────────────────────────────────────────────
y_pred_scaled  = model.predict(X_test_w,  verbose=0).flatten()
y_pred_c       = inverse_target(y_pred_scaled, scaler, n_features)
y_test_c       = inverse_target(y_test_w,      scaler, n_features)

# Train predictions — used only for overfitting check
y_train_pred_s = model.predict(X_train_w, verbose=0).flatten()
y_train_pred_c = inverse_target(y_train_pred_s, scaler, n_features)
y_train_c      = inverse_target(y_train_w,      scaler, n_features)


def compute_metrics(actual, pred, label):
    rmse  = np.sqrt(mean_squared_error(actual, pred))
    mae   = mean_absolute_error(actual, pred)
    r2    = r2_score(actual, pred)
    smape = np.mean(2 * np.abs(pred - actual) /
                    (np.abs(actual) + np.abs(pred) + 1e-8)) * 100
    print(f"  {label}")
    print(f"    RMSE  : {rmse:.4f} °C")
    print(f"    MAE   : {mae:.4f} °C")
    print(f"    R²    : {r2:.4f}")
    print(f"    sMAPE : {smape:.2f}%")
    return rmse, mae, r2


print("\n" + "=" * 62)
print("  LSTM TEST SET METRICS")
print("=" * 62)
rmse_test,  mae_test,  r2_test  = compute_metrics(y_test_c,  y_pred_c,       "Test")
rmse_train, mae_train, r2_train = compute_metrics(y_train_c, y_train_pred_c, "Train (overfit check)")

# Overfit warning
overfit_ratio = rmse_test / (rmse_train + 1e-9)
if overfit_ratio > 3.0:
    print(f"\n  ⚠ WARNING: Test RMSE is {overfit_ratio:.1f}x Train RMSE — model may be overfitting.")
    print("    Consider reducing model size or increasing dropout.")
else:
    print(f"\n  ✓ Overfit ratio (Test/Train RMSE): {overfit_ratio:.2f}  (healthy if < 3.0)")

# ── CELL 10: PREDICTION PLOT ──────────────────────────────────────────────────
# test_dates_aligned: the date axis for the test predictions.
# Slice combined_df dates starting from index len(train_df) to get test dates.
test_dates_aligned = combined_df[DATE_COL].values[
    len(train_df) : len(train_df) + len(X_test_w)
]

os.makedirs('reports', exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates_aligned, y_test_c,  label='Actual',    color='steelblue', lw=1.8)
ax.plot(test_dates_aligned, y_pred_c,  label='LSTM Pred', color='tomato',    lw=1.8, ls='--')
ax.set_title(f'LSTM — Actual vs Predicted  (RMSE={rmse_test:.3f}°C  R²={r2_test:.4f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved -> reports/lstm_actual_vs_predicted.png")

# Loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Train Loss', color='steelblue', lw=1.8)
ax.plot(history.history['val_loss'], label='Val Loss',   color='tomato',    lw=1.8, ls='--')
ax.axvline(best_epoch, color='green', ls=':', lw=1.5, label=f'Best epoch ({best_epoch})')
ax.set_title('LSTM Training Loss Curve', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss')
ax.legend()
plt.tight_layout()
plt.savefig('reports/lstm_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved -> reports/lstm_loss_curve.png")

# ── CELL 11: SAVE PREDICTIONS FOR ENSEMBLE ────────────────────────────────────
#
# FIX: Original stored date as numpy datetime64 — this caused occasional
# format mismatches when 06_ensemble.py merged on 'date' with the XGB/LGB CSVs.
# Now explicitly converting to YYYY-MM-DD string before saving.
#
# ALIGNMENT:
#   test_dates_aligned has exactly len(test_df) dates (114 rows).
#   id = 0..113 matches all other model prediction files for ensemble merge.

os.makedirs('data/predictions', exist_ok=True)

pred_df = pd.DataFrame({
    'date'      : pd.to_datetime(test_dates_aligned).strftime('%Y-%m-%d'),  # FIX
    'id'        : range(len(y_pred_c)),
    'prediction': y_pred_c,
    'actual'    : y_test_c
})

pred_df.to_csv('data/predictions/lstm.csv', index=False)
print(f"\n  Saved -> data/predictions/lstm.csv  ({len(pred_df)} rows)")
print(f"  Date range: {pred_df['date'].iloc[0]} -> {pred_df['date'].iloc[-1]}")

# ── CELL 12: SAVE MODEL ───────────────────────────────────────────────────────
model.save('models/lstm_model.keras')
print("  Saved -> models/lstm_model.keras")

# ── CELL 13: SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  LSTM PIPELINE COMPLETE")
print("=" * 62)
print(f"  Architecture : BiLSTM(128) + BiLSTM(64) + MHA(4h,k32) + LSTM(32)")
print(f"  Parameters   : {model.count_params():,}")
print(f"  Lookback     : {LOOKBACK} days")
print(f"  Features     : {n_features}  (includes cyclical time + cleaned data)")
print(f"  Epochs run   : {len(history.history['loss'])}")
print(f"  Best epoch   : {best_epoch}")
print(f"  Test RMSE    : {rmse_test:.4f} °C")
print(f"  Test R²      : {r2_test:.4f}")
print(f"  Predictions  : data/predictions/lstm.csv  ({len(pred_df)} rows)")
print("=" * 62)
print("\n  Next step -> run 05_arima_model.py, then 06_ensemble.py")
