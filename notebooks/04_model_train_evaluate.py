# PHASE 5–9: MODEL TRAINING, EVALUATION, SHAP, ENSEMBLE, SAVING

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
print(f"Test labels available: {y_test is not None}")

# ── CELL 2: Evaluation Helper ─────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    ss   = 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_true.mean())**2))
    print(f"{'─'*40}")
    print(f"Model : {name}")
    print(f"RMSE  : {rmse:.4f} °C")
    print(f"MAE   : {mae:.4f} °C")
    print(f"MAPE  : {mape:.2f}%")
    print(f"R²    : {ss:.4f}")
    print(f"{'─'*40}")
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': ss}

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

# ── TEST SET EVALUATION (XGBoost) ─────────────────────────────
if y_test is not None:
    xgb_test_results = evaluate("XGBoost [TEST]", y_test, xgb_test_pred)
    print("  ↑ XGBoost evaluated on HELD-OUT TEST SET")
else:
    xgb_test_results = None
    print("  NOTE: y_test not available — test metrics skipped.")

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

# ── TEST SET EVALUATION (LightGBM) ────────────────────────────
if y_test is not None:
    lgb_test_results = evaluate("LightGBM [TEST]", y_test, lgb_test_pred)
    print("  ↑ LightGBM evaluated on HELD-OUT TEST SET")
else:
    lgb_test_results = None
    print("  NOTE: y_test not available — test metrics skipped.")

# ── CELL 6: ACTUAL vs PREDICTED PLOTS ─────────────────────────
# --- 6a: VALIDATION plots (unchanged) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Validation Set — Actual vs Predicted", fontsize=13, fontweight='bold')
val_dates = train_fe['date'].iloc[split_idx:].values
for ax, name, pred in zip(axes,
                           ['XGBoost', 'LightGBM'],
                           [xgb_val_pred, lgb_val_pred]):
    ax.plot(val_dates, y_val.values, label='Actual',    color='steelblue',  lw=1.5)
    ax.plot(val_dates, pred,         label='Predicted', color='tomato',     lw=1.5, ls='--')
    ax.set_title(f'{name} – Actual vs Predicted (Validation)', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.legend()
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted_val.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/actual_vs_predicted_val.png")

# --- 6b: TEST SET plots (NEW) ---
if y_test is not None:
    test_dates = test_fe['date'].values
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Test Set — Actual vs Predicted", fontsize=13, fontweight='bold')
    for ax, name, pred in zip(axes,
                               ['XGBoost', 'LightGBM'],
                               [xgb_test_pred, lgb_test_pred]):
        ax.plot(test_dates, y_test.values, label='Actual',    color='steelblue',  lw=1.5)
        ax.plot(test_dates, pred,          label='Predicted', color='tomato',     lw=1.5, ls='--')
        ax.set_title(f'{name} – Actual vs Predicted (Test)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Mean Temperature (°C)')
        ax.legend()
    plt.tight_layout()
    plt.savefig("reports/actual_vs_predicted_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> reports/actual_vs_predicted_test.png")
else:
    print("  TEST PLOT SKIPPED: y_test not available.")

# ── CELL 7: MODEL COMPARISON TABLE ────────────────────────────
print("\n=== VALIDATION MODEL COMPARISON ===")
comparison_df = pd.DataFrame([xgb_results, lgb_results])
print(comparison_df.to_string(index=False))

if xgb_test_results and lgb_test_results:
    print("\n=== TEST SET MODEL COMPARISON ===")
    test_comparison_df = pd.DataFrame([xgb_test_results, lgb_test_results])
    print(test_comparison_df.to_string(index=False))

# Bar charts — validation
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Validation Set — Model Comparison", fontsize=12, fontweight='bold')
for ax, metric in zip(axes, ['RMSE', 'MAE', 'MAPE', 'R2']):
    bars = ax.bar(comparison_df['model'], comparison_df[metric],
                  color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
    ax.set_title(f'{metric} Comparison (Val)', fontsize=12)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("reports/model_comparison_val.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/model_comparison_val.png")

# Bar charts — test set (NEW)
if xgb_test_results and lgb_test_results:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Test Set — Model Comparison", fontsize=12, fontweight='bold')
    for ax, metric in zip(axes, ['RMSE', 'MAE', 'MAPE', 'R2']):
        bars = ax.bar(test_comparison_df['model'], test_comparison_df[metric],
                      color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
        ax.set_title(f'{metric} Comparison (Test)', fontsize=12)
        ax.set_ylabel(metric)
        for bar, val in zip(bars, test_comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/model_comparison_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> reports/model_comparison_test.png")

# ── CELL 8: RESIDUAL ANALYSIS ─────────────────────────────────
# --- 8a: Validation residuals (unchanged) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Validation Residuals", fontsize=12, fontweight='bold')
for ax, name, pred in zip(axes,
                           ['XGBoost', 'LightGBM'],
                           [xgb_val_pred, lgb_val_pred]):
    residuals = y_val.values - pred
    ax.scatter(pred, residuals, alpha=0.4, color='steelblue', s=15)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    ax.set_title(f'{name} Residuals (Validation)', fontsize=12)
    ax.set_xlabel('Predicted Temperature (°C)')
    ax.set_ylabel('Residual (Actual − Predicted)')
plt.tight_layout()
plt.savefig("reports/residuals_val.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/residuals_val.png")

# --- 8b: Test residuals (NEW) ---
if y_test is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Test Set Residuals", fontsize=12, fontweight='bold')
    for ax, name, pred in zip(axes,
                               ['XGBoost', 'LightGBM'],
                               [xgb_test_pred, lgb_test_pred]):
        residuals = y_test.values - pred
        ax.scatter(pred, residuals, alpha=0.4, color='darkorange', s=15)
        ax.axhline(0, color='red', lw=1.5, ls='--')
        ax.set_title(f'{name} Residuals (Test)', fontsize=12)
        ax.set_xlabel('Predicted Temperature (°C)')
        ax.set_ylabel('Residual (Actual − Predicted)')
    plt.tight_layout()
    plt.savefig("reports/residuals_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> reports/residuals_test.png")
else:
    print("  TEST RESIDUALS SKIPPED: y_test not available.")

# ── CELL 9: SHAP EXPLAINABILITY ───────────────────────────────
# SHAP on VALIDATION (standard for explainability — represents
# distribution seen during training; test set SHAP also added below).
print("\n  Computing SHAP values (XGBoost) on Validation…")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values   = explainer_xgb.shap_values(X_val)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_val, plot_type="bar",
                  max_display=15, show=False)
plt.title("XGBoost – SHAP Feature Importance (Validation)", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/shap_bar.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val, max_display=15, show=False)
plt.title("XGBoost – SHAP Beeswarm Plot (Validation)", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/shap_beeswarm.png")

# SHAP on TEST SET (NEW)
print("\n  Computing SHAP values (XGBoost) on Test Set…")
shap_values_test = explainer_xgb.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_test, X_test, plot_type="bar",
                  max_display=15, show=False)
plt.title("XGBoost – SHAP Feature Importance (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_bar_test.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/shap_bar_test.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_test, X_test, max_display=15, show=False)
plt.title("XGBoost – SHAP Beeswarm Plot (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_beeswarm_test.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/shap_beeswarm_test.png")

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
Weights are computed from VALIDATION RMSE (standard practice —
test set must remain strictly unseen during model selection).
"""

w_xgb = 1 / xgb_results['RMSE']
w_lgb = 1 / lgb_results['RMSE']
total = w_xgb + w_lgb
w_xgb /= total
w_lgb /= total

print(f"\nEnsemble weights – XGBoost: {w_xgb:.3f}, LightGBM: {w_lgb:.3f}")
print("  (weights derived from validation RMSE — test set remains unseen)")

# Ensemble on validation
ensemble_val_pred  = w_xgb * xgb_val_pred  + w_lgb * lgb_val_pred
ens_results = evaluate("Ensemble (Weighted Avg)", y_val, ensemble_val_pred)

# Ensemble on TEST (NEW)
ensemble_test_pred = w_xgb * xgb_test_pred + w_lgb * lgb_test_pred

if y_test is not None:
    ens_test_results = evaluate("Ensemble (Weighted Avg) [TEST]", y_test, ensemble_test_pred)
    print("  ↑ Ensemble evaluated on HELD-OUT TEST SET")
else:
    ens_test_results = None

# Final comparison — validation
final_df = pd.DataFrame([xgb_results, lgb_results, ens_results])
print("\n=== FINAL MODEL COMPARISON (Validation) ===")
print(final_df.to_string(index=False))

# Final comparison — test (NEW)
if xgb_test_results and lgb_test_results and ens_test_results:
    final_test_df = pd.DataFrame([xgb_test_results, lgb_test_results, ens_test_results])
    print("\n=== FINAL MODEL COMPARISON (Test Set) ===")
    print(final_test_df.to_string(index=False))

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
os.makedirs("data/predictions", exist_ok=True)

test_dates_col = test_fe['date'].values
actual_col     = y_test.values if y_test is not None else [None] * len(xgb_test_pred)

xgb_pred_df = pd.DataFrame({
    'date'      : test_dates_col,
    'id'        : range(len(xgb_test_pred)),
    'prediction': xgb_test_pred,        # ← TEST SET predictions
    'actual'    : actual_col
})
xgb_pred_df.to_csv("data/predictions/xgb.csv", index=False)

lgb_pred_df = pd.DataFrame({
    'date'      : test_dates_col,
    'id'        : range(len(lgb_test_pred)),
    'prediction': lgb_test_pred,        # ← TEST SET predictions
    'actual'    : actual_col
})
lgb_pred_df.to_csv("data/predictions/lgb.csv", index=False)

# Also save ensemble test predictions (NEW)
ens_pred_df = pd.DataFrame({
    'date'      : test_dates_col,
    'id'        : range(len(ensemble_test_pred)),
    'prediction': ensemble_test_pred,   # ← TEST SET predictions
    'actual'    : actual_col
})
ens_pred_df.to_csv("data/predictions/ensemble_xgb_lgb.csv", index=False)

print(f"\n Prediction CSVs saved (all on TEST SET):")
print(f"   data/predictions/xgb.csv  ({len(xgb_pred_df)} rows)")
print(f"   data/predictions/lgb.csv  ({len(lgb_pred_df)} rows)")
print(f"   data/predictions/ensemble_xgb_lgb.csv  ({len(ens_pred_df)} rows)")
print(f"   Date range: {pd.Timestamp(test_dates_col[0]).date()} -> {pd.Timestamp(test_dates_col[-1]).date()}")

# ── CELL 14: VAL vs TEST COMPARISON SUMMARY (NEW) ─────────────
print("\n" + "=" * 55)
print("  VAL vs TEST PERFORMANCE SUMMARY")
print("=" * 55)
rows = []
for val_r, test_r in [(xgb_results, xgb_test_results),
                       (lgb_results, lgb_test_results),
                       (ens_results, ens_test_results)]:
    model_name = val_r['model'].replace(' [TEST]', '')
    row = {
        'model'     : model_name,
        'val_RMSE'  : round(val_r['RMSE'], 4),
        'test_RMSE' : round(test_r['RMSE'], 4) if test_r else None,
        'val_MAE'   : round(val_r['MAE'],   4),
        'test_MAE'  : round(test_r['MAE'],  4) if test_r else None,
        'val_MAPE'  : round(val_r['MAPE'],  2) if 'MAPE' in val_r else None,
        'test_MAPE' : round(test_r['MAPE'], 2) if test_r and 'MAPE' in test_r else None,
        'val_R2'    : round(val_r['R2'],   4),
        'test_R2'   : round(test_r['R2'],  4) if test_r else None,
    }
    rows.append(row)

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
print("=" * 55)
print("  NOTE: A large val↔test gap signals overfitting.")
print("        Small gap = model generalises well.")
