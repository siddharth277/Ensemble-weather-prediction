import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import shap

# ── CELL 1: Load Data ─────────────────────────────────────────
train_fe = pd.read_csv("data/processed/train_features.csv", parse_dates=['date'])
test_fe  = pd.read_csv("data/processed/test_features.csv",  parse_dates=['date'])
meta     = joblib.load("models/feature_meta.pkl")

FEATURES = meta['features']
TARGET   = meta['target']

X_train = train_fe[FEATURES]
y_train = train_fe[TARGET]
X_test  = test_fe[FEATURES].fillna(test_fe[FEATURES].median())
y_test  = test_fe[TARGET] if TARGET in test_fe.columns else None

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── CELL 2: Evaluation Helper ─────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss   = 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_true.mean())**2))
    print(f"{'─'*40}")
    print(f"Model : {name}")
    print(f"RMSE  : {rmse:.4f} °C")
    print(f"MAE   : {mae:.4f} °C")
    print(f"R²    : {ss:.4f}")
    print(f"{'─'*40}")
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': ss}

# ── CELL 3: TIME-SERIES CROSS VALIDATION ──────────────────────
# IMPORTANT: Use TimeSeriesSplit – never shuffle for time series!
tscv = TimeSeriesSplit(n_splits=5)

# ── CELL 4: XGBOOST MODEL ─────────────────────────────────────
"""
XGBoost Hyperparameter Notes
─────────────────────────────
n_estimators    : number of trees – more = better but slower
max_depth       : depth of each tree – deeper = more complex patterns
learning_rate   : step size – smaller = slower but more precise
subsample       : % of rows used per tree – prevents overfitting
colsample_bytree: % of features used per tree – prevents overfitting
"""

xgb_model = xgb.XGBRegressor(
    n_estimators      = 500,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    early_stopping_rounds = 30,
    eval_metric       = 'rmse'
)

# Train/val split (last 20% as validation – respects time order)
split_idx = int(len(X_train) * 0.8)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50
)
xgb_val_pred  = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)
xgb_results = evaluate("XGBoost", y_val, xgb_val_pred)

# ── CELL 5: LIGHTGBM MODEL ────────────────────────────────────
"""
LightGBM Hyperparameter Notes
──────────────────────────────
num_leaves   : max leaves per tree – larger = more complex
min_data_in_leaf : min samples per leaf – prevents overfitting on small groups
feature_fraction : % of features per iteration (like colsample_bytree)
bagging_fraction : % of data per iteration (like subsample)
"""

lgb_model = lgb.LGBMRegressor(
    n_estimators      = 500,
    num_leaves        = 31,
    learning_rate     = 0.05,
    min_data_in_leaf  = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False),
               lgb.log_evaluation(50)]
)

lgb_val_pred  = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)
lgb_results = evaluate("LightGBM", y_val, lgb_val_pred)
# ── CELL 6: ACTUAL vs PREDICTED PLOTS ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
val_dates = train_fe['date'].iloc[split_idx:].values
for ax, name, pred in zip(axes,
                           ['XGBoost', 'LightGBM'],
                           [xgb_val_pred, lgb_val_pred]):
    ax.plot(val_dates, y_val.values, label='Actual',    color='steelblue',  lw=1.5)
    ax.plot(val_dates, pred,         label='Predicted', color='tomato',     lw=1.5, ls='--')
    ax.set_title(f'{name} – Actual vs Predicted', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.legend()
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 7: MODEL COMPARISON TABLE ────────────────────────────
comparison_df = pd.DataFrame([xgb_results, lgb_results])
print("\n=== MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    bars = ax.bar(comparison_df['model'], comparison_df[metric],
                  color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
    ax.set_title(f'{metric} Comparison', fontsize=12)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 8: RESIDUAL ANALYSIS ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, name, pred in zip(axes,
                           ['XGBoost', 'LightGBM'],
                           [xgb_val_pred, lgb_val_pred]):
    residuals = y_val.values - pred
    ax.scatter(pred, residuals, alpha=0.4, color='steelblue', s=15)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    ax.set_title(f'{name} Residuals', fontsize=12)
    ax.set_xlabel('Predicted Temperature (°C)')
    ax.set_ylabel('Residual (Actual − Predicted)')
plt.tight_layout()
plt.savefig("reports/residuals.png", dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 9: SHAP EXPLAINABILITY ───────────────────────────────
print("\n  Computing SHAP values (XGBoost)…")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values   = explainer_xgb.shap_values(X_val)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_val, plot_type="bar",
                  max_display=15, show=False)
plt.title("XGBoost – SHAP Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val, max_display=15, show=False)
plt.title("XGBoost – SHAP Beeswarm Plot", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()

"""
HOW TO INTERPRET SHAP
──────────────────────
- Each dot = one prediction
- Red = high feature value, Blue = low
- X-axis = SHAP value = how much that feature pushed prediction up/down
- temp_lag1 at the top means: yesterday's temperature is the biggest driver.
  If it was high (red), it pushes today's prediction higher (positive SHAP).
"""

# ── CELL 10: ENSEMBLE MODEL ───────────────────────────────────
"""
WEIGHTED AVERAGING ENSEMBLE
────────────────────────────
We give more weight to the model with lower RMSE.
Weight formula: w_i = (1/RMSE_i) / sum(1/RMSE_j)
This way, the more accurate model contributes more.
"""

w_xgb = 1 / xgb_results['RMSE']
w_lgb = 1 / lgb_results['RMSE']
total = w_xgb + w_lgb
w_xgb /= total
w_lgb /= total

print(f"\nEnsemble weights – XGBoost: {w_xgb:.3f}, LightGBM: {w_lgb:.3f}")

ensemble_val_pred  = w_xgb * xgb_val_pred  + w_lgb * lgb_val_pred
ensemble_test_pred = w_xgb * xgb_test_pred + w_lgb * lgb_test_pred

ens_results = evaluate("Ensemble (Weighted Avg)", y_val, ensemble_val_pred)

# Final comparison
final_df = pd.DataFrame([xgb_results, lgb_results, ens_results])
print("\n=== FINAL MODEL COMPARISON ===")
print(final_df.to_string(index=False))

# ── CELL 11: SAVE MODELS ──────────────────────────────────────
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(lgb_model, "models/lightgbm_model.pkl")
joblib.dump({'w_xgb': w_xgb, 'w_lgb': w_lgb}, "models/ensemble_weights.pkl")

print("\n Models saved:")
print("   models/xgboost_model.pkl")
print("   models/lightgbm_model.pkl")
print("   models/ensemble_weights.pkl")
print("   models/feature_meta.pkl")

# ── CELL 12: Verify Models Load Correctly ─────────────────────
xgb_loaded = joblib.load("models/xgboost_model.pkl")
sample = X_test.iloc[:3]
print("\n Sanity check – Predictions on first 3 test rows:")
print(xgb_loaded.predict(sample))

# ── CELL 13: SAVE TEST PREDICTIONS FOR ENSEMBLE ───────────────
# Saves standardised prediction CSVs so 06_ensemble.py can merge
# all models by (date, id).  Each file: date, id, prediction, actual.
import os
os.makedirs("data/predictions", exist_ok=True)

test_dates_col = test_fe['date'].values
actual_col     = y_test.values if y_test is not None else [None] * len(xgb_test_pred)

xgb_pred_df = pd.DataFrame({
    'date'      : test_dates_col,
    'id'        : range(len(xgb_test_pred)),
    'prediction': xgb_test_pred,
    'actual'    : actual_col
})
xgb_pred_df.to_csv("data/predictions/xgb.csv", index=False)

lgb_pred_df = pd.DataFrame({
    'date'      : test_dates_col,
    'id'        : range(len(lgb_test_pred)),
    'prediction': lgb_test_pred,
    'actual'    : actual_col
})
lgb_pred_df.to_csv("data/predictions/lgb.csv", index=False)

print(f"\n Prediction CSVs saved:")
print(f"   data/predictions/xgb.csv  ({len(xgb_pred_df)} rows)")
print(f"   data/predictions/lgb.csv  ({len(lgb_pred_df)} rows)")
print(f"   Date range: {pd.Timestamp(test_dates_col[0]).date()} -> {pd.Timestamp(test_dates_col[-1]).date()}")
