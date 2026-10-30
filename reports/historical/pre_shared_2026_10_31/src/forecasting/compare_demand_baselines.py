
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "demand_baseline_comparison.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

if df["Timestamp"].diff().dropna().ne(
    pd.Timedelta(hours=1)
).any():
    raise ValueError("Expected a continuous hourly time series.")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
test = df.iloc[val_end:].copy()

# Baseline 1: demand from the same hour on the previous day.
test["Seasonal Naive"] = df["Demand (MW)"].shift(24).iloc[val_end:].to_numpy()

# Baseline 2: constant mean calculated from training data only.
training_mean = train["Demand (MW)"].mean()
test["Training Mean"] = training_mean

actual = test["Demand (MW)"].to_numpy()

rows = []

for model_name in ["Seasonal Naive", "Training Mean"]:
    predicted = test[model_name].to_numpy()
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    rows.append({
        "Model": model_name,
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "MAPE (%)": mape,
    })

results = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("GRIDCASTAI — BASELINE COMPARISON")
print(f"Training-only mean: {training_mean:.2f} MW")
print(results.round(2).to_string(index=False))
print(f"\nSaved to: {OUTPUT_PATH}")
