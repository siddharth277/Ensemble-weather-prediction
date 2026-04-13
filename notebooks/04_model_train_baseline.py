
# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5–9 (BASELINE): MODEL TRAINING & EVALUATION — ORIGINAL FEATURES ONLY
# ══════════════════════════════════════════════════════════════════════════════
#
# PIPELINE TYPE : BASELINE
# PURPOSE       : Train XGBoost and LightGBM using ONLY the 3 original
#                 features (humidity, wind_speed, meanpressure).
#                 No engineered features — this is the lower-bound benchmark.
#
# WHY THIS MATTERS?
# ─────────────────
# Without a baseline, we can't say "feature engineering improved R² by X".
# This file trains the same model architectures as the advanced pipeline,
# but on raw features. The performance gap between the two pipelines is the
# quantified value of feature engineering.
#
# KEY DESIGN CHOICES (same as advanced):
#   - Time-based 80/20 train/val split (no shuffling — respects time order)
#   - TimeSeriesSplit cross-validation
#   - Same evaluation metrics: RMSE, MAE, R²
#   - Weighted ensemble (XGB + LGB)
#   - SHAP explainability
#
# DIFFERENCES FROM ADVANCED (04_model_train_evaluate.py):
#   - Loads train_baseline.csv (NOT train_features.csv)
#   - Uses 3 features (NOT 40+)
#   - No early_stopping_rounds (fewer features = less overfitting risk)
#   - Saves to xgb_baseline.pkl / lgb_baseline.pkl (NOT xgboost_model.pkl)
#
# INPUT  : data/processed/train_baseline.csv
#          data/processed/test_baseline.csv
#          models/baseline_feature_meta.pkl
# OUTPUT : models/xgb_baseline.pkl
#          models/lgb_baseline.pkl
#          models/baseline_ensemble_weights.pkl
#          reports/baseline_actual_vs_predicted.png
#          reports/baseline_model_comparison.png
#          reports/baseline_residuals.png
#          reports/baseline_shap.png
# ══════════════════════════════════════════════════════════════════════════════

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

# ── CELL 1: Load Baseline Data ────────────────────────────────────────────────
# Load from train_baseline.csv — NOT train_features.csv.
# train_baseline contains only the original cleaned columns.
# train_features contains 40+ engineered columns (advanced pipeline).

train_bl = pd.read_csv("data/processed/train_baseline.csv", parse_dates=['date'])
test_bl  = pd.read_csv("data/processed/test_baseline.csv",  parse_dates=['date'])

# Load feature list from metadata saved by 03_feature_engineering_baseline.py
meta     = joblib.load("models/baseline_feature_meta.pkl")
FEATURES = meta['features']    # ['humidity', 'wind_speed', 'meanpressure']
TARGET   = meta['target']      # 'meantemp'

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

# ── CELL 2: Evaluation Helper ─────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    """Compute RMSE, MAE, R² and pretty-print results."""
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

# ── CELL 3: Time-Series Train / Validation Split ──────────────────────────────
# IMPORTANT: Always use a chronological split for time-series data.
# Shuffling would leak future information into training → inflated metrics.

tscv      = TimeSeriesSplit(n_splits=5)
split_idx = int(len(X_train) * 0.80)    # last 20% as held-out validation

X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

print(f"\n  Time-based split:")
print(f"    Train set : {len(X_tr)} rows  ({split_idx} / {len(X_train)} = 80%)")
print(f"    Val set   : {len(X_val)} rows  (last 20% in time)")
print(f"    Val dates : {train_bl['date'].iloc[split_idx].date()} "
      f"→ {train_bl['date'].iloc[-1].date()}")

# ── CELL 4: XGBOOST BASELINE ──────────────────────────────────────────────────
#
# HYPERPARAMETER NOTES (BASELINE TUNING):
# ────────────────────────────────────────
# Fewer features → less capacity needed → smaller trees (max_depth=4 vs 5).
# No early_stopping_rounds because with only 3 features the model is
# much less likely to overfit, and keeping it simple aids reproducibility.
# n_estimators=300 (vs 500 advanced) — adequate for simpler feature space.

print(f"\n  Training XGBoost (Baseline)...")

xgb_model = xgb.XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,          # shallower: 3 features don't need deep trees
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

# ── CELL 5: LIGHTGBM BASELINE ─────────────────────────────────────────────────
#
# HYPERPARAMETER NOTES (BASELINE TUNING):
# ────────────────────────────────────────
# Same rationale — simpler feature space gets simpler model config.
# num_leaves=15 (vs 31 advanced) — avoids over-segmentation on 3 features.

print(f"\n  Training LightGBM (Baseline)...")

lgb_model = lgb.LGBMRegressor(
    n_estimators     = 300,
    num_leaves       = 15,         # smaller: 3 features have limited splits
    learning_rate    = 0.05,
    min_data_in_leaf = 20,
    feature_fraction = 1.0,        # use all 3 features every iteration
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

# ── CELL 6: ACTUAL vs PREDICTED PLOT ──────────────────────────────────────────
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

# ── CELL 7: MODEL COMPARISON TABLE ────────────────────────────────────────────
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

# ── CELL 8: RESIDUAL ANALYSIS ─────────────────────────────────────────────────
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

# ── CELL 9: SHAP EXPLAINABILITY ───────────────────────────────────────────────
# SHAP shows which of the 3 raw features matter most.
# Compare this to the advanced pipeline's SHAP — temp_lag1 will dominate there,
# while here we expect humidity and meanpressure to share importance.
print("\n  Computing SHAP values (XGBoost Baseline)...")

explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values   = explainer_xgb.shap_values(X_val)

plt.figure(figsize=(8, 4))
shap.summary_plot(shap_values, X_val, plot_type="bar",
                  max_display=len(FEATURES), show=False)
plt.title("Baseline XGBoost — SHAP Feature Importance\n"
          "(all 3 raw features shown)", fontsize=12)
plt.tight_layout()
plt.savefig("reports/baseline_shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_shap_bar.png")

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_val, max_display=len(FEATURES), show=False)
plt.title("Baseline XGBoost — SHAP Beeswarm\n"
          "(contrast with advanced pipeline)", fontsize=12)
plt.tight_layout()
plt.savefig("reports/baseline_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_shap_beeswarm.png")

# ── CELL 10: WEIGHTED ENSEMBLE ────────────────────────────────────────────────
# Same ensemble strategy as advanced pipeline:
#   Weight = (1 / RMSE) / sum(1 / RMSE_i)
# The lower-error model contributes more to the final prediction.

w_xgb = 1 / xgb_results['RMSE']
w_lgb = 1 / lgb_results['RMSE']
total = w_xgb + w_lgb
w_xgb /= total
w_lgb /= total

print(f"\n  Ensemble weights — XGBoost: {w_xgb:.3f}  LightGBM: {w_lgb:.3f}")

ensemble_val_pred  = w_xgb * xgb_val_pred  + w_lgb * lgb_val_pred
ensemble_test_pred = w_xgb * xgb_test_pred + w_lgb * lgb_test_pred

ens_results = evaluate("Ensemble (Baseline)", y_val, ensemble_val_pred)

# Full comparison including ensemble
final_df = pd.DataFrame([xgb_results, lgb_results, ens_results])
print("\n=== FULL BASELINE COMPARISON (including ensemble) ===")
print(final_df.to_string(index=False))

# ── CELL 11: SAVE BASELINE MODELS ─────────────────────────────────────────────
# Models saved with '_baseline' suffix — completely separate from advanced
# models (xgboost_model.pkl / lightgbm_model.pkl). Both coexist safely.

os.makedirs("models", exist_ok=True)

joblib.dump(xgb_model,                               "models/xgb_baseline.pkl")
joblib.dump(lgb_model,                               "models/lgb_baseline.pkl")
joblib.dump({'w_xgb': w_xgb, 'w_lgb': w_lgb},       "models/baseline_ensemble_weights.pkl")

print("\n  Baseline models saved:")
print("    models/xgb_baseline.pkl")
print("    models/lgb_baseline.pkl")
print("    models/baseline_ensemble_weights.pkl")

# ── CELL 12: Verify Models Load Correctly ─────────────────────────────────────
xgb_loaded = joblib.load("models/xgb_baseline.pkl")
sample_pred = xgb_loaded.predict(X_test.iloc[:3])
print(f"\n  Sanity check — Predictions on first 3 test rows:")
print(f"  {sample_pred.round(2)} °C")
print("  Model loads and predicts correctly.")

# ── CELL 13: Final Summary ────────────────────────────────────────────────────
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
