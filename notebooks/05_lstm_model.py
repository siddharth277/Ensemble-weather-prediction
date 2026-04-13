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

                                                                                
                                                       
                                                                         

train_df = pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COL])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=[DATE_COL])

train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
test_df  = test_df.sort_values(DATE_COL).reset_index(drop=True)

                                                                            
dupes = test_df.duplicated(subset=DATE_COL, keep=False).sum()
if dupes > 0:
    print(f'  Duplicate dates in test: {dupes} — keeping last')
    test_df = test_df.drop_duplicates(subset=DATE_COL, keep='last').reset_index(drop=True)

print(f"\n  Train : {len(train_df)} rows  "
      f"{train_df[DATE_COL].iloc[0].date()} -> {train_df[DATE_COL].iloc[-1].date()}")
print(f"  Test  : {len(test_df)} rows   "
      f"{test_df[DATE_COL].iloc[0].date()} -> {test_df[DATE_COL].iloc[-1].date()}")

                                                                                
 
                                  
                                    
                                                                                
                                                                           
                                                                             
                                                                 
 
                                                                         
                                                                           

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

                                                                              
                                                                    
    roll_mean_30       = df['meantemp'].rolling(30).mean()
    roll_std_30        = df['meantemp'].rolling(30).std().replace(0, 1)
    df['temp_zscore_30'] = (df['meantemp'] - roll_mean_30) / roll_std_30

                                                                              
    df['humidity_diff_1']    = df['humidity'].diff(1)
    df['humidity_roll_7']    = df['humidity'].rolling(7).mean()
    df['humidity_roll_std_7']= df['humidity'].rolling(7).std()

                                                                              
    df['wind_roll_7']  = df['wind_speed'].rolling(7).mean()
    df['wind_diff_1']  = df['wind_speed'].diff(1)

                                                                              
    df['pressure_diff_1'] = df['meanpressure'].diff(1)
    df['pressure_diff_3'] = df['meanpressure'].diff(3)
    df['pressure_roll_7'] = df['meanpressure'].rolling(7).mean()

                                                                              
    df['humidity_x_wind']  = df['humidity']     * df['wind_speed']
    df['temp_x_humidity']  = df['meantemp']      * df['humidity']
    df['temp_x_wind']      = df['meantemp']      * df['wind_speed']
    df['wind_x_pressure']  = df['wind_speed']    * df['meanpressure']

                                                                          
    df = df.ffill().fillna(0)

    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

print(f"\n  Features after engineering : {len(train_df.columns)} columns")

                                                                                
 
                                                                 
                                                                             
                                                                       

feature_cols = [c for c in train_df.columns if c != DATE_COL]
                                                                  
feature_cols = [TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]
n_features   = len(feature_cols)

print(f"  Total input features       : {n_features}")

                                                                   
combined_df     = pd.concat([train_df, test_df], ignore_index=True)                    .sort_values(DATE_COL).reset_index(drop=True)

scaler          = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[feature_cols])                                
combined_scaled = scaler.transform(combined_df[feature_cols])

                                                                                      
def create_sliding_window(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])                     
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = create_sliding_window(combined_scaled, LOOKBACK)

                                                          
split_idx        = len(train_df) - LOOKBACK
X_train_w, y_train_w = X_all[:split_idx], y_all[:split_idx]
X_test_w,  y_test_w  = X_all[split_idx:], y_all[split_idx:]

print(f"  X_train windows            : {X_train_w.shape}")
print(f"  X_test  windows            : {X_test_w.shape}  "
      f"(should match {len(test_df)} test rows)")

                                                                                
 
                         
                                                                                
                                                                                        
                                                                                       
                                                                       
                                                        

n_lookback = X_train_w.shape[1]

inputs = Input(shape=(n_lookback, n_features), name='input')

                
x = Bidirectional(LSTM(256, return_sequences=True), name='bilstm_1')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

                           
x2 = Bidirectional(LSTM(128, return_sequences=True), name='bilstm_2')(x)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.10)(x2)

                                                                     
skip = Dense(256, use_bias=False, name='skip_proj')(x)
x    = Add(name='residual_add')([x2, skip])
x    = LayerNormalization(name='layer_norm')(x)

                                     
attn_out = MultiHeadAttention(
    num_heads=4, key_dim=64, dropout=0.1, name='mha'
)(x, x)
context = GlobalAveragePooling1D(name='attn_pool')(attn_out)                 

                                 
seq_out = LSTM(64, return_sequences=False, name='lstm_final')(x)
seq_out = Dropout(0.10)(seq_out)

                     
merged = Concatenate(name='merge')([seq_out, context])                       

                       
z = Dense(128, activation='relu', name='dense_1')(merged)
z = Dropout(0.10)(z)
z = Dense(64,  activation='relu', name='dense_2')(z)
z = Dense(32,  activation='relu', name='dense_3')(z)
outputs = Dense(1, name='output')(z)

model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_MHA_v3')

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer = optimizer,
    loss      = tf.keras.losses.Huber(delta=1.0),                      
    metrics   = ['mae']
)

print(f"\n  Model parameters : {model.count_params():,}")

                                                                                
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=35,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=10,
        min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        filepath='models/lstm_best.keras',
        monitor='val_loss', save_best_only=True, verbose=0
    )
]

print(f"\n  Training (up to {EPOCHS} epochs, early stopping at patience=35)...")
history = model.fit(
    X_train_w, y_train_w,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_split= VAL_SPLIT,
    callbacks       = callbacks,
    verbose         = 1
)

best_epoch   = int(np.argmin(history.history['val_loss']))
best_val_loss = min(history.history['val_loss'])
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Best val_loss : {best_val_loss:.4f}")

                                                                               
def inverse_target(scaled_values, scaler, n_feat):
    

       
    dummy       = np.zeros((len(scaled_values), n_feat))
    dummy[:, 0] = scaled_values
    return scaler.inverse_transform(dummy)[:, 0]

                                                                                
y_pred_scaled = model.predict(X_test_w, verbose=0).flatten()
y_pred_c      = inverse_target(y_pred_scaled, scaler, n_features)
y_test_c      = inverse_target(y_test_w,      scaler, n_features)

                                       
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
rmse_test, mae_test, r2_test = compute_metrics(y_test_c,  y_pred_c,      "Test")
rmse_train, mae_train, r2_train = compute_metrics(y_train_c, y_train_pred_c, "Train (overfit check)")

                                                                                
                                                    
test_dates_aligned = combined_df[DATE_COL].values[
    len(train_df) : len(train_df) + len(X_test_w)
]

os.makedirs('reports', exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates_aligned, y_test_c,  label='Actual',    color='steelblue',  lw=1.8)
ax.plot(test_dates_aligned, y_pred_c,  label='LSTM Pred', color='tomato',     lw=1.8, ls='--')
ax.set_title(f'LSTM — Actual vs Predicted  (RMSE={rmse_test:.3f}°C  R²={r2_test:.4f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved -> reports/lstm_actual_vs_predicted.png")

            
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

                                                                                
 
                      
                                                                 
                                                                 
                                                                  

os.makedirs('data/predictions', exist_ok=True)

pred_df = pd.DataFrame({
                : test_dates_aligned,
                : range(len(y_pred_c)),
                : y_pred_c,
                : y_test_c
})

pred_df.to_csv('data/predictions/lstm.csv', index=False)
print(f"\n  Saved -> data/predictions/lstm.csv  ({len(pred_df)} rows)")
print(f"  Date range: {pred_df.date.iloc[0]} -> {pred_df.date.iloc[-1]}")

import joblib

model.save('models/lstm_model.keras')
print("  Saved -> models/lstm_model.keras")

joblib.dump(scaler, 'models/lstm_scaler.pkl')
print("  Saved -> models/lstm_scaler.pkl")

joblib.dump(feature_cols, 'models/lstm_features.pkl')
print("  Saved -> models/lstm_features.pkl")

                                                                                
print("\n" + "=" * 62)
print("  LSTM PIPELINE COMPLETE")
print("=" * 62)
print(f"  Architecture : BiLSTM(256) + BiLSTM(128) + MHA + LSTM(64)")
print(f"  Lookback     : {LOOKBACK} days")
print(f"  Features     : {n_features}")
print(f"  Epochs run   : {len(history.history['loss'])}")
print(f"  Best epoch   : {best_epoch}")
print(f"  Test RMSE    : {rmse_test:.4f} °C")
print(f"  Test R²      : {r2_test:.4f}")
print(f"  Predictions  : data/predictions/lstm.csv  ({len(pred_df)} rows)")
print("=" * 62)
print("\n  Next step -> run 05_arima_model.py, then 06_ensemble.py")
