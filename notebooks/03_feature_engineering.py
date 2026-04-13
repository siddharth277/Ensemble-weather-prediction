import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

                                                                
train = pd.read_csv("data/processed/train_clean.csv", parse_dates=['date'])
test  = pd.read_csv("data/processed/test_clean.csv",  parse_dates=['date'])

print(f"Train: {train.shape}, Test: {test.shape}")

                                                                

   

def engineer_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)

                                                                 
    df['year']        = df['date'].dt.year
    df['month']       = df['date'].dt.month
    df['day']         = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek

                                                    
    df['season'] = df['month'].map({
        12: 1, 1: 1, 2: 1,
        3: 2,  4: 2, 5: 2,
        6: 3,  7: 3, 8: 3,
        9: 4, 10: 4, 11: 4
    })

                                                              
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

                                                                
    for lag in [1, 2, 3, 7]:
        df[f'temp_lag{lag}']      = df['meantemp'].shift(lag)
        df[f'humidity_lag{lag}']  = df['humidity'].shift(lag)
        df[f'pressure_lag{lag}']  = df['meanpressure'].shift(lag)
        df[f'wind_lag{lag}']      = df['wind_speed'].shift(lag)

                                                                
    for window in [3, 7, 14]:
        df[f'temp_roll_mean{window}'] = df['meantemp'].shift(1).rolling(window).mean()
        df[f'temp_roll_std{window}']  = df['meantemp'].shift(1).rolling(window).std()
        df[f'hum_roll_mean{window}']  = df['humidity'].shift(1).rolling(window).mean()

                                                           
    df['temp_ewm7']  = df['meantemp'].shift(1).ewm(span=7).mean()
    df['temp_ewm14'] = df['meantemp'].shift(1).ewm(span=14).mean()

                                                                
    df['heat_index']     = df['meantemp'] * df['humidity'] / 100
    df['pressure_delta'] = df['meanpressure'] - df['meanpressure'].shift(1)
    df['temp_delta']     = df['meantemp'] - df['meantemp'].shift(1)
    df['wind_chill']     = df['meantemp'] - 0.5 * df['wind_speed']

    return df

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

print(f"\nFeatures before engineering: {train.shape[1]}")
print(f"Features after  engineering: {train_fe.shape[1]}")
print("\nNew feature columns:")
new_cols = [c for c in train_fe.columns if c not in train.columns]
for c in new_cols:
    print(f"  {c}")

                                                                
                                                      
train_fe = train_fe.dropna().reset_index(drop=True)

TARGET   = 'meantemp'
DROP_COLS = ['date', 'year']                                                  
FEATURES = [c for c in train_fe.columns if c not in [TARGET] + DROP_COLS]

X_train = train_fe[FEATURES]
y_train = train_fe[TARGET]

                                                         
X_test  = test_fe[FEATURES].fillna(test_fe[FEATURES].median())
y_test  = test_fe[TARGET] if TARGET in test_fe.columns else None

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_train stats : mean={y_train.mean():.2f}, std={y_train.std():.2f}")

                                                                
corr_with_target = X_train.join(y_train).corr()[TARGET].drop(TARGET).sort_values(
    key=abs, ascending=False
)

plt.figure(figsize=(10, 6))
corr_with_target.head(20).plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Top 20 Feature Correlations with meantemp', fontsize=13)
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.savefig("reports/feature_importance_corr.png", dpi=150, bbox_inches='tight')
plt.show()

                                                                
train_fe.to_csv("data/processed/train_features.csv", index=False)
test_fe.to_csv("data/processed/test_features.csv",   index=False)

import joblib
feature_meta = {'features': FEATURES, 'target': TARGET}
joblib.dump(feature_meta, "models/feature_meta.pkl")

print("\n Feature-engineered data saved!")
print(f"   Features list saved to models/feature_meta.pkl")
