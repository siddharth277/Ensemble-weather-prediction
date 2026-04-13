import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import shap

print("=" * 60)
print("  BASELINE MODEL TRAINING — Original Features Only")
print("=" * 60)

                                                                                
                                                        
                                                            
                                                                     

train_bl = pd.read_csv("data/processed/train_baseline.csv", parse_dates=['date'])
test_bl  = pd.read_csv("data/processed/test_baseline.csv",  parse_dates=['date'])

                                                                             
meta     = joblib.load("models/baseline_feature_meta.pkl")
FEATURES = meta['features']                                                
TARGET   = meta['target']                  

X_train = train_bl[FEATURES]
y_train = train_bl[TARGET]
X_test  = test_bl[FEATURES].fillna(test_bl[FEATURES].median())
y_test  = test_bl[TARGET] if TARGET in test_bl.columns else None

print(f"\n  Pipeline   : {meta['pipeline']}")
print(f"  Features   : {len(FEATURES)}  ->  {FEATURES}")
print(f"  X_train    : {X_train.shape}")
print(f"  X_test     : {X_test.shape}")
print(f"  y mean     : {y_train.mean():.2f}°C   std: {y_train.std():.2f}°C")
print(f"\n  ADVANCED pipeline uses 40+ features for comparison.")

                                                                                
def evaluate(name, y_true, y_pred):
                                                         
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_true.mean())**2))
    print(f"{'─' * 42}")
    print(f"  Model : {name}")
    print(f"  RMSE  : {rmse:.4f} °C")
    print(f"  MAE   : {mae:.4f} °C")
    print(f"  R²    : {r2:.4f}")
    print(f"{'─' * 42}")
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

                                                                                
                                                                   
                                                                           

tscv      = TimeSeriesSplit(n_splits=5)
split_idx = int(len(X_train) * 0.80)                                     

X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

print(f"\n  Time-based split:")
print(f"    Train set : {len(X_tr)} rows  ({split_idx} / {len(X_train)} = 80%)")
print(f"    Val set   : {len(X_val)} rows  (last 20% in time)")
print(f"    Val dates : {train_bl['date'].iloc[split_idx].date()} "
      f"→ {train_bl['date'].iloc[-1].date()}")

                                                                                
 
                                         
                                          
                                                                           
                                                                    
                                                                          
                                                                          

print(f"\n  Training XGBoost (Baseline)...")

xgb_model = xgb.XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,                                                       
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    eval_metric      = 'rmse',
    verbosity        = 0
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_val_pred  = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)
xgb_results   = evaluate("XGBoost (Baseline)", y_val, xgb_val_pred)

                                                                                
 
                                         
                                          
                                                                   
                                                                          

print(f"\n  Training LightGBM (Baseline)...")

lgb_model = lgb.LGBMRegressor(
    n_estimators     = 300,
    num_leaves       = 15,                                                  
    learning_rate    = 0.05,
    min_data_in_leaf = 20,
    feature_fraction = 1.0,                                            
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False),
               lgb.log_evaluation(-1)]
)

lgb_val_pred  = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)
lgb_results   = evaluate("LightGBM (Baseline)", y_val, lgb_val_pred)

                                                                                
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("BASELINE — Actual vs Predicted Temperature", fontsize=14, fontweight='bold')
val_dates = train_bl['date'].iloc[split_idx:].values

for ax, name, pred in zip(axes,
                           ['XGBoost (Baseline)', 'LightGBM (Baseline)'],
                           [xgb_val_pred, lgb_val_pred]):
    ax.plot(val_dates, y_val.values, label='Actual',    color='steelblue', lw=1.5)
    ax.plot(val_dates, pred,         label='Predicted', color='tomato',    lw=1.5, ls='--')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.legend()

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/baseline_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_actual_vs_predicted.png")

                                                                                
comparison_df = pd.DataFrame([xgb_results, lgb_results])
print("\n=== BASELINE MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Baseline Model Comparison", fontsize=13, fontweight='bold')

for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    bars = ax.bar(
        [m.replace(' (Baseline)', '') for m in comparison_df['model']],
        comparison_df[metric],
        color=['#4C72B0', '#DD8452'],
        edgecolor='white',
        width=0.5
    )
    ax.set_title(f'{metric}', fontsize=12)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("reports/baseline_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_model_comparison.png")

                                                                                
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Baseline — Residual Analysis", fontsize=13, fontweight='bold')

for ax, name, pred in zip(axes,
                           ['XGBoost (Baseline)', 'LightGBM (Baseline)'],
                           [xgb_val_pred, lgb_val_pred]):
    residuals = y_val.values - pred
    ax.scatter(pred, residuals, alpha=0.4, color='steelblue', s=15)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    ax.set_title(f'{name} Residuals', fontsize=11)
    ax.set_xlabel('Predicted Temperature (°C)')
    ax.set_ylabel('Residual (Actual − Predicted)')

plt.tight_layout()
plt.savefig("reports/baseline_residuals.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_residuals.png")

                                                                                
                                                     
                                                                               
                                                                     
print("\n  Computing SHAP values (XGBoost Baseline)...")

explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values   = explainer_xgb.shap_values(X_val)

plt.figure(figsize=(8, 4))
shap.summary_plot(shap_values, X_val, plot_type="bar",
                  max_display=len(FEATURES), show=False)
plt.title("Baseline XGBoost — SHAP Feature Importance\n"
                                      , fontsize=12)
plt.tight_layout()
plt.savefig("reports/baseline_shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_shap_bar.png")

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_val, max_display=len(FEATURES), show=False)
plt.title("Baseline XGBoost — SHAP Beeswarm\n"
                                             , fontsize=12)
plt.tight_layout()
plt.savefig("reports/baseline_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_shap_beeswarm.png")

                                                                                
                                              
                                         
                                                                 

w_xgb = 1 / xgb_results['RMSE']
w_lgb = 1 / lgb_results['RMSE']
total = w_xgb + w_lgb
w_xgb /= total
w_lgb /= total

print(f"\n  Ensemble weights — XGBoost: {w_xgb:.3f}  LightGBM: {w_lgb:.3f}")

ensemble_val_pred  = w_xgb * xgb_val_pred  + w_lgb * lgb_val_pred
ensemble_test_pred = w_xgb * xgb_test_pred + w_lgb * lgb_test_pred

ens_results = evaluate("Ensemble (Baseline)", y_val, ensemble_val_pred)

                                    
final_df = pd.DataFrame([xgb_results, lgb_results, ens_results])
print("\n=== FULL BASELINE COMPARISON (including ensemble) ===")
print(final_df.to_string(index=False))

                                                                                
                                                                          
                                                                       

os.makedirs("models", exist_ok=True)

joblib.dump(xgb_model,                               "models/xgb_baseline.pkl")
joblib.dump(lgb_model,                               "models/lgb_baseline.pkl")
joblib.dump({'w_xgb': w_xgb, 'w_lgb': w_lgb},       "models/baseline_ensemble_weights.pkl")

print("\n  Baseline models saved:")
print("    models/xgb_baseline.pkl")
print("    models/lgb_baseline.pkl")
print("    models/baseline_ensemble_weights.pkl")

                                                                                
xgb_loaded = joblib.load("models/xgb_baseline.pkl")
sample_pred = xgb_loaded.predict(X_test.iloc[:3])
print(f"\n  Sanity check — Predictions on first 3 test rows:")
print(f"  {sample_pred.round(2)} °C")
print("  Model loads and predicts correctly.")

                                                                                
print("\n" + "=" * 60)
print("  BASELINE TRAINING COMPLETE — FINAL METRICS")
print("=" * 60)
print(f"  Pipeline  : BASELINE  (3 raw features)")
print(f"  XGBoost   RMSE={xgb_results['RMSE']:.4f}  MAE={xgb_results['MAE']:.4f}  R²={xgb_results['R2']:.4f}")
print(f"  LightGBM  RMSE={lgb_results['RMSE']:.4f}  MAE={lgb_results['MAE']:.4f}  R²={lgb_results['R2']:.4f}")
print(f"  Ensemble  RMSE={ens_results['RMSE']:.4f}  MAE={ens_results['MAE']:.4f}  R²={ens_results['R2']:.4f}")
print()
print("  Now run the ADVANCED pipeline to see the improvement:")
print("    03_feature_engineering.py  ->  04_model_train_evaluate.py")
print("  Compare R² and RMSE to quantify the value of feature engineering.")
print("=" * 60)
print("\n  Baseline model training complete!")
