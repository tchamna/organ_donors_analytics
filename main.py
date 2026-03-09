import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT
from lightgbm import LGBMRegressor

# 1. DATA PREPARATION (The "Nixtla" Format)
# All these models require three columns: unique_id, ds (datestamp), and y (target)
referal_path = r"sample_data\l1_bronze\referrals.csv"
df = pd.read_csv(referal_path)
df['ds'] = pd.to_datetime(df['referral_ts'], errors='coerce')
df = df.dropna(subset=['ds'])

# Aggregate to daily volume
data = df.groupby(df['ds'].dt.date).size().reset_index(name='y')
data['ds'] = pd.to_datetime(data['ds'])
data['unique_id'] = 'total_referrals' # Required for global models

# Split into Train and Test
horizon = 14  # Predict next 14 days
train = data[:-horizon]
test = data[-horizon:]

# 2. MLFORECAST (LightGBM)
# Excels at capturing tabular features and lags
mlf = MLForecast(
    models=[LGBMRegressor(n_estimators=100, verbosity=-1)],
    freq='D',
    lags=[1, 7, 14],
    target_transforms=[Differences([1])] # Stationarize the data
)
mlf.fit(train)
lgbm_preds = mlf.predict(horizon)

# 3. NEURALFORECAST (NHITS & TFT)
# NHITS: Hierarchical interpolation for long horizons
# TFT: Attention-based model that captures complex temporal dynamics
models = [
    NHITS(h=horizon, input_size=3*horizon, max_steps=100),
    TFT(h=horizon, input_size=3*horizon, max_steps=100)
]
nf = NeuralForecast(models=models, freq='D')
nf.fit(df=train)
neural_preds = nf.predict()

# 4. MERGE RESULTS & COMPARE
results = test.copy().set_index('ds')
results['LightGBM'] = lgbm_preds['LGBMRegressor'].values
results['NHITS'] = neural_preds['NHITS'].values
results['TFT'] = neural_preds['TFT'].values
results = results.reset_index()
plot_data = (
    train[['ds', 'y']].tail(30).rename(columns={'y': 'Historical (Last 30d)'})
    .merge(results, on='ds', how='outer')
    .sort_values('ds')
)
plot_data = plot_data[
    ['ds', 'Historical (Last 30d)', 'y', 'LightGBM', 'NHITS', 'TFT']
]

metrics = []
for model in ['LightGBM', 'NHITS', 'TFT']:
    mae = (results['y'] - results[model]).abs().mean()
    metrics.append({"model": model, "mae": round(float(mae), 4)})

# Export an offline artifact for the dashboard prediction tab.
output_dir = Path("sample_data") / "l3_gold"
output_dir.mkdir(parents=True, exist_ok=True)
plot_data.to_csv(output_dir / "gold_offline_model_forecast_results.csv", index=False)
pd.DataFrame(metrics).to_csv(output_dir / "gold_offline_model_forecast_metrics.csv", index=False)

# 5. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(train['ds'][-30:], train['y'][-30:], label='Historical (Last 30d)')
plt.plot(results['ds'], results['y'], label='Actual', color='black', linewidth=2)
plt.plot(results['ds'], results['LightGBM'], label='LightGBM', linestyle='--')
plt.plot(results['ds'], results['NHITS'], label='NHITS', linestyle='--')
plt.plot(results['ds'], results['TFT'], label='TFT', linestyle='--')
plt.title('SOTA Model Comparison: Daily Referral Forecasting')
plt.legend()
plt.show()

# Calculate Accuracy
for row in metrics:
    print(f"{row['model']} MAE: {row['mae']:.2f}")
