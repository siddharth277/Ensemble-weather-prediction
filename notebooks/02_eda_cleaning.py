import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("Libraries loaded successfully")

                                                                
train = pd.read_csv("data/raw/Train.csv")
test  = pd.read_csv("data/raw/Test.csv")

train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
train.head()

                                                                

   
print(train.dtypes)
print("\nBasic stats:")
train.describe()

                                                                 
print("=== MISSING VALUES ===")
print(train.isnull().sum())
print(f"\nTotal missing: {train.isnull().sum().sum()}")

                                                                  
print(f"Duplicate rows: {train.duplicated().sum()}")

                                                                
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(train['meantemp'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Mean Temperature (°C)', fontsize=13)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Count')

             
axes[1].plot(train['date'], train['meantemp'], color='steelblue', alpha=0.8, linewidth=0.8)
axes[1].set_title('Mean Temperature Over Time', fontsize=13)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°C)')

plt.tight_layout()
plt.savefig("reports/eda_temperature.png", dpi=150, bbox_inches='tight')
plt.show()

                                                                
numeric_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
corr = train[numeric_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig("reports/eda_correlation.png", dpi=150, bbox_inches='tight')
plt.show()

   

                                                                 
train['month'] = train['date'].dt.month
monthly = train.groupby('month')['meantemp'].mean()

plt.figure(figsize=(10, 4))
monthly.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Average Monthly Temperature (Delhi, 2013–2017)', fontsize=13)
plt.xlabel('Month')
plt.ylabel('Avg Temperature (°C)')
plt.xticks(range(12), ['Jan','Feb','Mar','Apr','May','Jun',
                             'Aug','Sep','Oct','Nov','Dec'], rotation=0)
plt.tight_layout()
plt.savefig("reports/eda_monthly.png", dpi=150, bbox_inches='tight')
plt.show()

                                                                
print("=== OUTLIER DETECTION ===")
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

for col in features:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = train[(train[col] < lower) | (train[col] > upper)]
    print(f"{col:15s} | Q1={Q1:8.2f} Q3={Q3:8.2f} | "
          f"Range=[{lower:.2f}, {upper:.2f}] | Outliers={len(outliers)}")

                                                                       

                                                                
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, features):
    ax.boxplot(train[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.set_title(col)
plt.suptitle('Boxplots – Outlier Visualisation', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("reports/eda_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()

                                                                 

   

def clean_weather_data(df):
    df = df.copy()

                          
    mask_pressure = (df['meanpressure'] < 900) | (df['meanpressure'] > 1100)
    print(f"Fixing {mask_pressure.sum()} bad pressure rows")
    df.loc[mask_pressure, 'meanpressure'] = np.nan
    df['meanpressure'] = df['meanpressure'].fillna(
        df['meanpressure'].rolling(3, min_periods=1, center=True).median()
    )

                        
    mask_wind = df['wind_speed'] < 0
    print(f"Fixing {mask_wind.sum()} negative wind rows")
    df.loc[mask_wind, 'wind_speed'] = np.nan

                      
    mask_hum = df['humidity'] > 100
    print(f"Capping {mask_hum.sum()} humidity rows > 100")
    df.loc[mask_hum, 'humidity'] = 100.0

                         
    numeric_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df

train_clean = clean_weather_data(train.copy())
test_clean  = clean_weather_data(test.copy())

                                                                
print("\n=== BEFORE vs AFTER CLEANING ===")
print(f"\nmeanpressure  BEFORE: min={train['meanpressure'].min():.2f}, "
      f"max={train['meanpressure'].max():.2f}")
print(f"meanpressure  AFTER : min={train_clean['meanpressure'].min():.2f}, "
      f"max={train_clean['meanpressure'].max():.2f}")

print(f"\nMissing values BEFORE: {train.isnull().sum().sum()}")
print(f"Missing values AFTER : {train_clean.isnull().sum().sum()}")

                                                                
train_clean.to_csv("data/processed/train_clean.csv", index=False)
test_clean.to_csv("data/processed/test_clean.csv",  index=False)
print("\n✅ Cleaned data saved to data/processed/")
