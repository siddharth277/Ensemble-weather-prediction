# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12: ENSEMBLE — Combine All Model Predictions Into One Final Forecast
# ══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Loads prediction CSVs from every model, merges on date, builds a weighted
#   average ensemble, evaluates all models side-by-side, and saves the final
#   combined forecast.
#
# PREREQUISITES (run in this order before this script):
#   1.  02_eda_cleaning.py
#   2a. 03_feature_engineering.py  →  04_model_train_evaluate.py   (XGB, LGB)
#   2b. 05_lstm_model.py                                            (LSTM)
#   2c. 05_arima_model.py                                           (AR, ARIMA, SARIMA)
#
# INPUT  : data/predictions/xgb.csv
#          data/predictions/lgb.csv
#          data/predictions/lstm.csv
#          data/predictions/arima.csv
#          data/predictions/sarima.csv
#          (ar.csv is also loaded if present, for reference)
#
# OUTPUT : data/predictions/ensemble_final.csv
#          reports/ensemble_all_models.png
#          reports/ensemble_comparison_bar.png
#
# PREDICTION FILE FORMAT (all models must produce this schema):
#   date        : YYYY-MM-DD  — the test date being predicted
#   id          : int 0..N-1  — row index for deterministic merging
#   prediction  : float       — predicted meantemp in °C
#   actual      : float       — observed meantemp in °C (for evaluation)
#
# ALIGNMENT GUARANTEE:
#   All models predict on the same 114 test dates (2017-01-01 → 2017-04-24).
#   The inner join on 'date' in Cell 3 ensures only common dates are kept,
#   so the ensemble is never contaminated by misaligned rows.
#
# ENSEMBLE STRATEGY:
#   Weighted average — higher weight to models with lower historical RMSE.
#   Default weights are a reasonable starting point; tune them in the
#   WEIGHTS dict below after inspecting each model's individual RMSE.
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive — safe for scripts and Colab
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 65)
print("  ENSEMBLE — Combining All Model Predictions")
print("=" * 65)

# ── CELL 1: ENSEMBLE CONFIG ───────────────────────────────────────────────────
#
# WEIGHTS — how much each model contributes to the final prediction.
# Rules:
#   • Must sum to 1.0
#   • Increase weight for models with lower test RMSE
#   • If a model file is missing it is skipped and weights are renormalised
#     automatically — the ensemble still runs with whatever is available.
#
# Starting rationale:
#   XGB / LGB : 0.25 each — advanced feature engineering gives ~R²=0.98
#   LSTM      : 0.20      — captures nonlinear temporal patterns, ~R²=0.95
#   ARIMA     : 0.15      — solid statistical baseline, ~R²=0.90
#   SARIMA    : 0.15      — adds seasonal component on top of ARIMA
# Total = 1.00

WEIGHTS = {
    'xgb'   : 0.25,
    'lgb'   : 0.25,
    'lstm'  : 0.20,
    'arima' : 0.15,
    'sarima': 0.15,
}

PRED_DIR    = 'data/predictions'
REPORT_DIR  = 'reports'
OUTPUT_FILE = 'data/predictions/ensemble_final.csv'

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, \
    f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values()):.4f}"

# ── CELL 2: LOAD PREDICTION FILES ────────────────────────────────────────────
#
# Each CSV is expected to have columns: date, id, prediction, actual.
# The 'actual' column must be identical across all files (same test set).
# Missing files are skipped with a warning — ensemble degrades gracefully.

print(f"\n  Loading prediction files from '{PRED_DIR}/'...")
print(f"  {'─'*50}")

frames  = {}           # model_name -> DataFrame
missing = []           # models whose file was not found

for model_name in WEIGHTS:
    path = os.path.join(PRED_DIR, f'{model_name}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Validate required columns
        required = {'date', 'prediction', 'actual'}
        if not required.issubset(df.columns):
            print(f"  SKIP {model_name:8s}: missing columns "
                  f"{required - set(df.columns)}")
            missing.append(model_name)
            continue

        frames[model_name] = df
        print(f"  OK   {model_name:8s} : {len(df):3d} rows  "
              f"{df.date.iloc[0].date()} → {df.date.iloc[-1].date()}  "
              f"pred_range=[{df.prediction.min():.1f}, {df.prediction.max():.1f}]°C")
    else:
        print(f"  MISS {model_name:8s}: {path} not found — skipping")
        missing.append(model_name)

if missing:
    print(f"\n  Excluded models : {missing}")

if len(frames) < 2:
    raise RuntimeError(
        f"Need at least 2 prediction files. Found only: {list(frames.keys())}.\n"
        f"Run the model scripts first:\n"
        f"  04_model_train_evaluate.py  →  xgb.csv, lgb.csv\n"
        f"  05_lstm_model.py            →  lstm.csv\n"
        f"  05_arima_model.py           →  arima.csv, sarima.csv"
    )

# ── CELL 3: RECALCULATE WEIGHTS FOR AVAILABLE MODELS ─────────────────────────
# If any model files were missing, renormalise remaining weights so they
# still sum to 1.0. This lets the ensemble work even during partial runs.

active_weights = {m: WEIGHTS[m] for m in frames}
total_w = sum(active_weights.values())
active_weights = {m: w / total_w for m, w in active_weights.items()}

print(f"\n  Active models and adjusted weights:")
for m, w in active_weights.items():
    bar = '█' * int(w * 40)
    print(f"    {m:8s}  {w:.3f}  {bar}")

# ── CELL 4: MERGE ALL PREDICTIONS ON DATE ────────────────────────────────────
#
# Strategy: inner join on 'date' column.
# All models should produce predictions for the same 114 test dates, so an
# inner join should retain all rows.  If a model has gaps (e.g. LSTM loses
# a few rows due to windowing), the join silently drops those dates — this
# is the correct behaviour (better than NaN-contaminated predictions).

print(f"\n  Merging predictions on 'date' (inner join)...")

merged  = None
actuals = None   # taken from the first loaded model — should be identical

for model_name, df in frames.items():
    pred_col = df[['date', 'prediction']].rename(
        columns={'prediction': f'pred_{model_name}'}
    )
    if merged is None:
        merged  = pred_col
        actuals = df[['date', 'actual']].copy()
    else:
        merged = merged.merge(pred_col, on='date', how='inner')

# Attach the actual values
merged = merged.merge(actuals, on='date', how='inner')
merged = merged.sort_values('date').reset_index(drop=True)
merged['id'] = range(len(merged))

print(f"  Merged shape : {merged.shape}")
print(f"  Date range   : {merged.date.iloc[0].date()} → {merged.date.iloc[-1].date()}")
print(f"  Rows dropped : {max(len(df) for df in frames.values()) - len(merged)}")

if len(merged) < 100:
    print(f"\n  WARNING: only {len(merged)} rows after merge.")
    print("  Check that all model scripts ran on the same test set.")

# ── CELL 5: BUILD ENSEMBLE PREDICTION ────────────────────────────────────────
# Weighted average: Σ(weight_i × prediction_i)

merged['prediction_ensemble'] = sum(
    w * merged[f'pred_{m}'] for m, w in active_weights.items()
)

print(f"\n  Ensemble prediction range : "
      f"[{merged.prediction_ensemble.min():.2f}, "
      f"{merged.prediction_ensemble.max():.2f}]°C")

# ── CELL 6: EVALUATE ALL MODELS + ENSEMBLE ───────────────────────────────────
actual = merged['actual'].values

def compute_metrics(actual, pred, label=''):
    rmse  = np.sqrt(mean_squared_error(actual, pred))
    mae   = mean_absolute_error(actual, pred)
    r2    = r2_score(actual, pred)
    mape  = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-8))) * 100
    if label:
        print(f"  {'─'*44}")
        print(f"  {label}")
        print(f"    RMSE  : {rmse:.4f} °C")
        print(f"    MAE   : {mae:.4f} °C")
        print(f"    MAPE  : {mape:.2f}%")
        print(f"    R²    : {r2:.4f}")
    return {'model': label, 'RMSE': round(rmse, 4),
            'MAE': round(mae, 4), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}

print("\n" + "=" * 65)
print("  INDIVIDUAL MODEL METRICS")
print("=" * 65)
results = []
for m in frames:
    r = compute_metrics(actual, merged[f'pred_{m}'].values, m.upper())
    results.append(r)

print("\n" + "=" * 65)
print("  ENSEMBLE METRICS")
print("=" * 65)
ens_result = compute_metrics(
    actual, merged['prediction_ensemble'].values, 'ENSEMBLE (weighted avg)'
)
results.append(ens_result)

# Sort by RMSE for easy reading
comparison = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
print("\n" + "=" * 65)
print("  FULL RANKING (sorted by RMSE ↑ = better)")
print("=" * 65)
print(comparison.to_string(index=False))

# Summary: did the ensemble beat all individuals?
best_individual_rmse  = comparison[
    comparison.model != 'ENSEMBLE (weighted avg)'
]['RMSE'].min()
ens_rmse = ens_result['RMSE']
delta    = best_individual_rmse - ens_rmse

print(f"\n  Best individual RMSE : {best_individual_rmse:.4f} °C")
print(f"  Ensemble RMSE        : {ens_rmse:.4f} °C")
if delta > 0:
    print(f"  Ensemble improvement : -{delta:.4f} °C  "
          f"({delta / best_individual_rmse * 100:.1f}% better)")
else:
    print(f"  Ensemble is {abs(delta):.4f} °C worse than best individual.")
    print("  Tip: reduce weight of weaker models in the WEIGHTS dict.")

# ── CELL 7: FULL FORECAST COMPARISON PLOT ────────────────────────────────────
os.makedirs(REPORT_DIR, exist_ok=True)

dates    = merged['date'].values
n_models = len(frames)
palette  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

fig, axes = plt.subplots(n_models + 1, 1,
                          figsize=(14, 4 * (n_models + 1)),
                          sharex=True)
fig.suptitle('All Models — Actual vs Predicted Temperature\n'
             '(Delhi 2017 Test Set)', fontsize=14, fontweight='bold')

# Individual model subplots
for ax, m, color in zip(axes[:n_models], list(frames.keys()), palette):
    pred   = merged[f'pred_{m}'].values
    r_row  = comparison[comparison.model == m.upper()].iloc[0]
    ax.plot(dates, actual, color='#333333', lw=1.8, label='Actual', zorder=3)
    ax.plot(dates, pred,   color=color, lw=1.5, ls='--',
            label=f'{m.upper()} Pred', zorder=2)
    ax.fill_between(dates, actual, pred, alpha=0.07, color=color)
    ax.set_title(
        f'{m.upper()}  |  RMSE={r_row.RMSE:.3f}°C  '
        f'MAE={r_row.MAE:.3f}°C  R²={r_row.R2:.4f}',
        fontsize=10, fontweight='bold'
    )
    ax.set_ylabel('Temp (°C)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

# Ensemble subplot (last)
ax_ens = axes[-1]
ens_r  = comparison[comparison.model == 'ENSEMBLE (weighted avg)'].iloc[0]
ax_ens.plot(dates, actual,
            color='#333333', lw=1.8, label='Actual', zorder=3)
ax_ens.plot(dates, merged['prediction_ensemble'].values,
            color='#2ca02c', lw=2.2, label='Ensemble', zorder=4)
ax_ens.fill_between(dates, actual, merged['prediction_ensemble'].values,
                    alpha=0.07, color='#2ca02c')
ax_ens.set_title(
    f'ENSEMBLE (weighted avg)  |  RMSE={ens_r.RMSE:.3f}°C  '
    f'MAE={ens_r.MAE:.3f}°C  R²={ens_r.R2:.4f}',
    fontsize=10, fontweight='bold'
)
ax_ens.set_ylabel('Temp (°C)')
ax_ens.set_xlabel('Date')
ax_ens.legend(fontsize=8, loc='upper left')
ax_ens.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'ensemble_all_models.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved → reports/ensemble_all_models.png")

# ── CELL 8: METRIC COMPARISON BAR CHART ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold')

bar_colors = ['#2ca02c' if m == 'ENSEMBLE (weighted avg)' else '#4C72B0'
              for m in comparison.model]
short_labels = [m.replace(' (weighted avg)', '\n(ensemble)')
                    .replace('ENSEMBLE', 'ENSEMBLE')
                for m in comparison.model]

for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    bars = ax.bar(
        range(len(comparison)), comparison[metric],
        color=bar_colors, edgecolor='white', width=0.6
    )
    ax.set_xticks(range(len(comparison)))
    ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
    ax.set_title(
        f'{metric}  ({"lower is better" if metric != "R2" else "higher is better"})',
        fontsize=11
    )
    ax.set_ylabel(metric)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, comparison[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(comparison[metric]) * 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold'
        )

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'ensemble_comparison_bar.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → reports/ensemble_comparison_bar.png")

# ── CELL 9: RESIDUAL PLOT ────────────────────────────────────────────────────
ens_residuals = actual - merged['prediction_ensemble'].values
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Ensemble Residual Analysis', fontsize=12, fontweight='bold')

axes[0].plot(dates, ens_residuals, color='#2ca02c', lw=1.2)
axes[0].axhline(0, color='red', lw=1.5, ls='--')
axes[0].fill_between(dates, ens_residuals, 0, alpha=0.15, color='#2ca02c')
axes[0].set_title('Residuals over Time', fontsize=10)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Error (Actual − Predicted) °C')
axes[0].grid(True, alpha=0.3)

axes[1].hist(ens_residuals, bins=25, color='#2ca02c', alpha=0.8, edgecolor='white')
axes[1].axvline(0,                      color='#333', lw=1.5, ls='--', label='Zero')
axes[1].axvline(ens_residuals.mean(),   color='red',  lw=1.5, ls='--',
                label=f'Mean={ens_residuals.mean():.2f}°C')
axes[1].set_title('Residual Distribution', fontsize=10)
axes[1].set_xlabel('Residual (°C)')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'ensemble_residuals.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → reports/ensemble_residuals.png")

# ── CELL 10: SAVE FINAL OUTPUT CSV ───────────────────────────────────────────
# Columns: date, id, actual, pred_<model>..., prediction_ensemble
output_cols = (
    ['date', 'id', 'actual']
    + [f'pred_{m}' for m in frames]
    + ['prediction_ensemble']
)
merged[output_cols].to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved → {OUTPUT_FILE}  ({len(merged)} rows)")
print(f"  Columns: {output_cols}")

# ── CELL 11: FINAL SUMMARY ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ENSEMBLE PIPELINE COMPLETE — FINAL SUMMARY")
print("=" * 65)
print(f"  Models combined  : {list(frames.keys())}")
if missing:
    print(f"  Models skipped   : {missing}")
print(f"  Test rows        : {len(merged)}")
print(f"  Date range       : "
      f"{merged.date.iloc[0].date()} → {merged.date.iloc[-1].date()}")
print()
print(f"  {'Model':<28}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}")
print(f"  {'─'*58}")
for _, row in comparison.iterrows():
    marker = ' ←' if row.model == 'ENSEMBLE (weighted avg)' else \
             ' ★' if row.RMSE == comparison.RMSE.min() else ''
    print(f"  {row.model:<28}  {row.RMSE:>8.4f}  {row.MAE:>8.4f}  {row.R2:>8.4f}{marker}")
print()
print("  Outputs:")
print(f"    data/predictions/ensemble_final.csv")
print(f"    reports/ensemble_all_models.png")
print(f"    reports/ensemble_comparison_bar.png")
print(f"    reports/ensemble_residuals.png")
print("=" * 65)
