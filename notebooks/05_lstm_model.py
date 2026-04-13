# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10: LSTM MODEL — Bidirectional LSTM + Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
LOOKBACK    = 30
EPOCHS      = 250
BATCH_SIZE  = 8
VAL_SPLIT   = 0.15

print("=" * 62)
print("  LSTM MODEL — Bidirectional LSTM + MultiHeadAttention")
print("=" * 62)

# ── CELL 2: LOAD DATA ─────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COL])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=[DATE_COL])

train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
test_df  = test_df.sort_values(DATE_COL).reset_index(drop=True)

dupes = test_df.duplicated(subset=DATE_COL, keep=False).sum()
if dupes > 0:
    test_df = test_df.drop_duplicates(subset=DATE_COL, keep='last').reset_index(drop=True)

# ── CELL 3: FEATURE ENGINEERING ───────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    df['temp_diff_1']  = df['meantemp'].diff(1)
    df['temp_diff_2']  = df['meantemp'].diff(2)
    df['temp_diff_3']  = df['meantemp'].diff(3)
    df['temp_diff_7']  = df['meantemp'].diff(7)
    df['temp_diff_14'] = df['meantemp'].diff(14)

    df['temp_accel'] = df['temp_diff_1'].diff(1)

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

    roll_mean_30 = df['meantemp'].rolling(30).mean()
    roll_std_30  = df['meantemp'].rolling(30).std().replace(0, 1)
    df['temp_zscore_30'] = (df['meantemp'] - roll_mean_30) / roll_std_30

    df['humidity_diff_1']     = df['humidity'].diff(1)
    df['humidity_roll_7']     = df['humidity'].rolling(7).mean()
    df['humidity_roll_std_7'] = df['humidity'].rolling(7).std()

    df['wind_roll_7'] = df['wind_speed'].rolling(7).mean()
    df['wind_diff_1'] = df['wind_speed'].diff(1)

    df['pressure_diff_1'] = df['meanpressure'].diff(1)
    df['pressure_diff_3'] = df['meanpressure'].diff(3)
    df['pressure_roll_7'] = df['meanpressure'].rolling(7).mean()

    df['humidity_x_wind'] = df['humidity'] * df['wind_speed']
    df['temp_x_humidity'] = df['meantemp'] * df['humidity']
    df['temp_x_wind']     = df['meantemp'] * df['wind_speed']
    df['wind_x_pressure'] = df['wind_speed'] * df['meanpressure']

    df = df.ffill().fillna(0)
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# ── CELL 4: SCALE & WINDOW ────────────────────────────────────────────────────
feature_cols = [c for c in train_df.columns if c != DATE_COL]
feature_cols = [TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]
n_features   = len(feature_cols)

combined_df = pd.concat([train_df, test_df], ignore_index=True)\
    .sort_values(DATE_COL).reset_index(drop=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[feature_cols])

# ✅ FIX: SAVE SCALER
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/lstm_scaler.pkl')
print("Saved lstm_scaler.pkl")

combined_scaled = scaler.transform(combined_df[feature_cols])

def create_sliding_window(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = create_sliding_window(combined_scaled, LOOKBACK)

split_idx = len(train_df) - LOOKBACK
X_train_w, y_train_w = X_all[:split_idx], y_all[:split_idx]
X_test_w,  y_test_w  = X_all[split_idx:], y_all[split_idx:]

# ── CELL 5: MODEL ─────────────────────────────────────────────────────────────
inputs = Input(shape=(LOOKBACK, n_features))

x = Bidirectional(LSTM(256, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x2 = Bidirectional(LSTM(128, return_sequences=True))(x)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.10)(x2)

skip = Dense(256, use_bias=False)(x)
x = Add()([x2, skip])
x = LayerNormalization()(x)

attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
context = GlobalAveragePooling1D()(attn_out)

seq_out = LSTM(64)(x)

merged = Concatenate()([seq_out, context])

z = Dense(128, activation='relu')(merged)
z = Dense(64, activation='relu')(z)
z = Dense(32, activation='relu')(z)
outputs = Dense(1)(z)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])

# ── CELL 6: TRAIN ─────────────────────────────────────────────────────────────
history = model.fit(
    X_train_w, y_train_w,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1
)

# ── CELL 7: SAVE MODEL ────────────────────────────────────────────────────────
model.save('models/lstm_model.keras')

print("\n✅ DONE — scaler + model saved")
