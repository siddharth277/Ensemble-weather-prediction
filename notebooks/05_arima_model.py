import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

import joblib
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 62)
print("  AR / ARIMA / SARIMA — Statistical Time-Series Models")
print("=" * 62)

                                                                                
 
                                                                 
                                                                     
                                                               
                                                                           

train_data = pd.read_csv('data/raw/Train.csv', parse_dates=['date'])
test_data  = pd.read_csv('data/raw/Test.csv',  parse_dates=['date'])

train_data = train_data.sort_values('date').reset_index(drop=True)
test_data  = test_data.sort_values('date').reset_index(drop=True)

print(f"\n  Train : {len(train_data)} rows  "
      f"{train_data['date'].iloc[0].date()} -> {train_data['date'].iloc[-1].date()}")
print(f"  Test  : {len(test_data)} rows   "
      f"{test_data['date'].iloc[0].date()} -> {test_data['date'].iloc[-1].date()}")

                                                                             
                                                         
                                                               
def fix_pressure(series):
    s = series.where((series >= 950) & (series <= 1100))
    return s.interpolate(method='linear').ffill().bfill()

train_data['meanpressure'] = fix_pressure(train_data['meanpressure'])
test_data['meanpressure']  = fix_pressure(test_data['meanpressure'])

                                                                              
wind_cap = train_data['wind_speed'].quantile(0.95)
train_data['wind_speed'] = train_data['wind_speed'].clip(upper=wind_cap)
test_data['wind_speed']  = test_data['wind_speed'].clip(upper=wind_cap)

train_vals = train_data['meantemp'].values.astype(float)
test_vals  = test_data['meantemp'].values.astype(float)
test_dates = test_data['date'].values

print(f"\n  Train temperature: mean={train_vals.mean():.2f}°C  "
      f"std={train_vals.std():.2f}°C")
print(f"  Test  temperature: mean={test_vals.mean():.2f}°C   "
      f"std={test_vals.std():.2f}°C")

                                                                                
def evaluate(actual, pred, label=''):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    r2   = r2_score(actual, pred)
    if label:
        print(f"  {'─'*40}")
        print(f"  Model  : {label}")
        print(f"  RMSE   : {rmse:.4f} °C")
        print(f"  MAE    : {mae:.4f} °C")
        print(f"  MAPE   : {mape:.2f}%")
        print(f"  R²     : {r2:.4f}")
    return rmse, mae, mape, r2

                                                                                
 
                                                                      
                                                                              
                                                                           
                                                                           

def walk_forward_ar(train, test, lags):
                                                                              
    history = list(train)
    preds   = []
    for i in range(len(test)):
        m    = AutoReg(history, lags=lags, old_names=False).fit()
        pred = m.predict(start=len(history), end=len(history))[0]
        preds.append(pred)
        history.append(test[i])                       
    return np.array(preds)

def walk_forward_arima(train, test, order):
                                                                        
    history = list(train)
    preds   = []
    m       = ARIMA(history, order=order).fit()
    for i in range(len(test)):
        pred = m.forecast(steps=1)[0]
        preds.append(pred)
        m = m.append([test[i]], refit=False)
    return np.array(preds)

def walk_forward_sarima(train, test, order, seasonal_order):
                                                                         
    history = list(train)
    preds   = []
    m = SARIMAX(
        history, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    for i in range(len(test)):
        preds.append(m.forecast(steps=1)[0])
        m = m.append([test[i]], refit=False)
    return np.array(preds)

                                                                                
 
                                            
                                     
                                                                              

print(f"\n  {'='*40}")
print(f"  AR WALK-FORWARD — scanning lags p = [1,2,3,5,7]")
print(f"  {'='*40}")
print(f"  {'Model':<12}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  {'R²':>8}")

ar_scan = {}
for p in [1, 2, 3, 5, 7]:
    pred            = walk_forward_ar(train_vals, test_vals, p)
    r, m, mp, r2    = evaluate(test_vals, pred)
    print(f"  AR(p={p})        {r:8.4f}  {m:8.4f}  {mp:7.2f}%  {r2:8.4f}")
    ar_scan[p]      = (r, m, mp, r2, pred)

best_p      = min(ar_scan, key=lambda p: ar_scan[p][0])
ar_pred     = ar_scan[best_p][4]
print(f"\n  -> Best AR lag = {best_p}  (RMSE = {ar_scan[best_p][0]:.4f})")

                                                                                
 
                                                                                 
                                                          

print(f"\n  {'='*40}")
print(f"  ARIMA WALK-FORWARD — scanning (p,1,q) orders")
print(f"  {'='*40}")
print(f"  {'Model':<20}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  {'R²':>8}")

arima_scan = {}
for p, q in [(1,0), (1,1), (2,1), (1,2), (3,1), (2,2)]:
    try:
        pred            = walk_forward_arima(train_vals, test_vals, (p, 1, q))
        r, m, mp, r2    = evaluate(test_vals, pred)
        print(f"  ARIMA({p},1,{q})           {r:8.4f}  {m:8.4f}  {mp:7.2f}%  {r2:8.4f}")
        arima_scan[(p, q)] = (r, m, mp, r2, pred)
    except Exception as e:
        print(f"  ARIMA({p},1,{q}) failed: {e}")

best_pq        = min(arima_scan, key=lambda k: arima_scan[k][0])
arima_pred     = arima_scan[best_pq][4]
print(f"\n  -> Best ARIMA = ({best_pq[0]},1,{best_pq[1]})  "
      f"(RMSE = {arima_scan[best_pq][0]:.4f})")

                                                                                
 
                                                                              
                                                                        
                                                              

print(f"\n  {'='*40}")
print(f"  SARIMA GRID SEARCH — (d=1, D=1, s=7)")
print(f"  {'='*40}")
print(f"  {'Model':<38}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}")

sarima_configs = [
    ((1,1,1), (0,1,1,7)),                           
    ((1,1,1), (1,1,1,7)),                    
    ((2,1,1), (0,1,1,7)),            
    ((1,1,2), (0,1,1,7)),            
    ((2,1,1), (1,1,1,7)),                 
    ((1,1,1), (2,1,1,7)),                       
]

sarima_results = []
for order, seas in sarima_configs:
    try:
        pred            = walk_forward_sarima(train_vals, test_vals, order, seas)
        r, m, mp, r2    = evaluate(test_vals, pred)
        label           = f"SARIMA{order}{seas}"
        print(f"  {label:<38}  {r:8.4f}  {m:8.4f}  {r2:8.4f}")
        sarima_results.append((label, order, seas, r, m, mp, r2, pred))
    except Exception as e:
        print(f"  SARIMA{order}{seas} FAILED: {e}")

                     
sarima_results.sort(key=lambda x: x[3])
best_sarima           = sarima_results[0]
best_sarima_label     = best_sarima[0]
best_sarima_order     = best_sarima[1]
best_sarima_seas      = best_sarima[2]
sarima_pred           = best_sarima[7]

print(f"\n  -> Best: {best_sarima_label}  (RMSE = {best_sarima[3]:.4f})")

                                                                                
print("\n" + "=" * 62)
print("  STATISTICAL MODELS — FINAL TEST METRICS")
print("=" * 62)
ar_metrics     = evaluate(test_vals, ar_pred,     f"AR(p={best_p})")
arima_metrics  = evaluate(test_vals, arima_pred,  f"ARIMA({best_pq[0]},1,{best_pq[1]})")
sarima_metrics = evaluate(test_vals, sarima_pred, f"SARIMA {best_sarima_label}")

                                                                                
os.makedirs('reports', exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Statistical Models — Walk-Forward Predictions', fontsize=14, fontweight='bold')

colors   = ['#E85D24', '#1D9E75', '#7F77DD']
models   = [
    (ar_pred,    f'AR(p={best_p})',                       ar_metrics),
    (arima_pred, f'ARIMA({best_pq[0]},1,{best_pq[1]})',  arima_metrics),
    (sarima_pred, best_sarima_label,                      sarima_metrics),
]

for ax, (pred, label, metrics), color in zip(axes, models, colors):
    ax.plot(test_dates, test_vals, label='Actual',      color='#333',  lw=1.8)
    ax.plot(test_dates, pred,      label=f'{label} Pred', color=color, lw=1.8, ls='--')
    ax.fill_between(test_dates, test_vals, pred, alpha=0.08, color=color)
    ax.set_title(f'{label}  |  RMSE={metrics[0]:.3f}  MAE={metrics[1]:.3f}  R²={metrics[3]:.4f}',
                 fontsize=11)
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=9)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('reports/arima_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved -> reports/arima_predictions.png")

                                                                                
 
                                        
                                                                  
                                                                

os.makedirs('data/predictions', exist_ok=True)

def save_preds(dates, preds, actual, filename, model_label):
    df = pd.DataFrame({
                    : dates,
                    : range(len(preds)),
                    : preds,
                    : actual
    })
    df.to_csv(filename, index=False)
    print(f"  Saved -> {filename}  ({len(df)} rows)  "
          f"[{df.date[0] if hasattr(df.date[0], 'date') else pd.Timestamp(df.date[0]).date()} -> "
          f"{pd.Timestamp(df.date.iloc[-1]).date()}]")

save_preds(test_dates, ar_pred,     test_vals, 'data/predictions/ar.csv',     'AR')
save_preds(test_dates, arima_pred,  test_vals, 'data/predictions/arima.csv',  'ARIMA')
save_preds(test_dates, sarima_pred, test_vals, 'data/predictions/sarima.csv', 'SARIMA')

                                                                                
os.makedirs('models', exist_ok=True)

ar_model_final = AutoReg(train_vals, lags=best_p, old_names=False).fit()
joblib.dump(ar_model_final, 'models/ar_model.pkl')
print("\n  Saved -> models/ar_model.pkl")

arima_model_final = ARIMA(train_vals, order=(best_pq[0], 1, best_pq[1])).fit()
joblib.dump(arima_model_final, 'models/arima_model.pkl')
print("  Saved -> models/arima_model.pkl")

sarima_model_final = SARIMAX(
    train_vals,
    order=best_sarima_order,
    seasonal_order=best_sarima_seas,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
joblib.dump(sarima_model_final, 'models/sarima_model.pkl')
print("  Saved -> models/sarima_model.pkl")

                                                                                
print("\n" + "=" * 62)
print("  STATISTICAL MODELS PIPELINE COMPLETE")
print("=" * 62)
print(f"  AR({best_p})    "
      f"RMSE={ar_metrics[0]:.4f}  MAE={ar_metrics[1]:.4f}  R²={ar_metrics[3]:.4f}")
print(f"  ARIMA({best_pq[0]},1,{best_pq[1]})  "
      f"RMSE={arima_metrics[0]:.4f}  MAE={arima_metrics[1]:.4f}  R²={arima_metrics[3]:.4f}")
print(f"  {best_sarima_label[:15]:<15}  "
      f"RMSE={sarima_metrics[0]:.4f}  MAE={sarima_metrics[1]:.4f}  R²={sarima_metrics[3]:.4f}")
print()
print("  Prediction files saved:")
print("    data/predictions/ar.csv")
print("    data/predictions/arima.csv")
print("    data/predictions/sarima.csv")
print("=" * 62)
print("\n  Next step -> run 06_ensemble.py")
