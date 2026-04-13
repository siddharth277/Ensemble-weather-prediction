import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

                                                                                
                                                              
train = pd.read_csv("data/processed/train_clean.csv", parse_dates=['date'])
test  = pd.read_csv("data/processed/test_clean.csv",  parse_dates=['date'])

print("=" * 55)
print("  BASELINE PIPELINE — Original Features Only")
print("=" * 55)
print(f"\n  Train shape : {train.shape}")
print(f"  Test shape  : {test.shape}")
print(f"\n  Columns in cleaned data:")
for col in train.columns:
    print(f"    {col}")

                                                                                
 
                                   
                                    
                                                   
                                               
                                                  
 
                                                         
                                                                   
 
                                                            
                                                                      
 
                         
                                                          
                                                             

TARGET   = 'meantemp'
FEATURES = ['humidity', 'wind_speed', 'meanpressure']

print(f"\n  Baseline feature set ({len(FEATURES)} features):")
for f in FEATURES:
    print(f"    {f}")
print(f"\n  Target: {TARGET}")
print(f"\n  (Advanced pipeline will use 40+ features for comparison)")

                                                                                
missing_in_train = [f for f in FEATURES + [TARGET] if f not in train.columns]
missing_in_test  = [f for f in FEATURES if f not in test.columns]

if missing_in_train:
    raise ValueError(f"Missing columns in train: {missing_in_train}")
if missing_in_test:
    raise ValueError(f"Missing columns in test: {missing_in_test}")

print("\n  All required columns present in both train and test.")

                                                                                
print("\n  Baseline feature statistics (train):")
print(train[FEATURES + [TARGET]].describe().round(3).to_string())

                                                                                
                                                                  
                                                                        
corr = train[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(
    key=abs, ascending=False
)

print(f"\n  Correlation of raw features with {TARGET}:")
for feat, val in corr.items():
    bar = "#" * int(abs(val) * 20)
    direction = "+" if val > 0 else "-"
    print(f"    {feat:<15}  {direction}{bar}  {val:+.4f}")

print("\n  Note: temp_lag1 (advanced pipeline) achieves ~0.98 correlation.")
print("    Raw features max out around 0.6 — that's the baseline ceiling.")

                                                                               
fig, axes = plt.subplots(1, len(FEATURES), figsize=(14, 4))
fig.suptitle("Baseline Feature Distributions (Train)", fontsize=13, fontweight='bold')

colors = ['#4C72B0', '#DD8452', '#55A868']
for ax, feat, color in zip(axes, FEATURES, colors):
    ax.hist(train[feat].dropna(), bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/baseline_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved -> reports/baseline_feature_distributions.png")

                                                                                
                                                                 
                                                                          

train_baseline = train.copy()
test_baseline  = test.copy()

                                                                       
for feat in FEATURES:
    median_val = train_baseline[feat].median()
    train_baseline[feat] = train_baseline[feat].fillna(median_val)
    test_baseline[feat]  = test_baseline[feat].fillna(median_val)

X_train = train_baseline[FEATURES]
y_train = train_baseline[TARGET]
X_test  = test_baseline[FEATURES]
y_test  = test_baseline[TARGET] if TARGET in test_baseline.columns else None

print(f"\n  X_train shape : {X_train.shape}  ({len(FEATURES)} features x {len(X_train)} rows)")
print(f"  X_test  shape : {X_test.shape}   ({len(FEATURES)} features x {len(X_test)} rows)")
print(f"  y_train stats : mean={y_train.mean():.2f}C, std={y_train.std():.2f}C")
print(f"\n  No NaN rows dropped (unlike advanced pipeline which drops ~14 rows for lags)")

                                                                                
                                                                   
                                                                   
os.makedirs("data/processed", exist_ok=True)

train_baseline.to_csv("data/processed/train_baseline.csv", index=False)
test_baseline.to_csv("data/processed/test_baseline.csv",   index=False)

print(f"\n  Saved -> data/processed/train_baseline.csv  ({len(train_baseline)} rows)")
print(f"  Saved -> data/processed/test_baseline.csv   ({len(test_baseline)} rows)")

                                                                               
                                                                    
                                                                           
os.makedirs("models", exist_ok=True)

baseline_meta = {
                 : FEATURES,
                 : TARGET,
                 : 'baseline',
                 : len(FEATURES),
                 : 'Original dataset features only — no engineering'
}
joblib.dump(baseline_meta, "models/baseline_feature_meta.pkl")

print(f"  Saved -> models/baseline_feature_meta.pkl")

                                                                                
print("\n" + "=" * 55)
print("  BASELINE FEATURE ENGINEERING — SUMMARY")
print("=" * 55)
print(f"  Pipeline     : BASELINE (no engineering)")
print(f"  Features     : {len(FEATURES)}  {FEATURES}")
print(f"  Train rows   : {len(train_baseline)}")
print(f"  Test rows    : {len(test_baseline)}")
print(f"  NaN rows     : 0  (none dropped)")
print()
print(f"  ADVANCED pipeline uses 40+ features and drops ~14 lag-NaN rows")
print("=" * 55)
print("\n  Baseline feature pipeline complete!")
print("     Next step -> run 04_model_train_baseline.py")
