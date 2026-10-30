
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "baseline_demand_predictions.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Verify that observations are consecutive hourly records.
intervals = df["Timestamp"].diff().dropna()
if not (intervals == pd.Timedelta(hours=1)).all():
    raise ValueError("Time series has missing or irregular hourly intervals.")

# Chronological split: first 70% train, next 15% validation, last 15% test.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

# Seasonal-naive forecast: use demand from 24 hours earlier.
combined = pd.concat([train, val, test], ignore_index=True)
combined["Forecast (MW)"] = combined["Demand (MW)"].shift(24)

test_predictions = combined.iloc[val_end:].copy()

# The first 24 observations of the test set use actual demand
# from the preceding day, which is available at forecast time.
actual = test_predictions["Demand (MW)"]
predicted = test_predictions["Forecast (MW)"]

mask = actual.notna() & predicted.notna()
actual = actual[mask]
predicted = predicted[mask]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = (
    np.abs((actual - predicted) / actual).mean() * 100
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results = test_predictions.loc[mask, ["Timestamp", "Demand (MW)"]].copy()
results["Forecast (MW)"] = predicted.to_numpy()
results["Absolute Error (MW)"] = np.abs(
    actual.to_numpy() - predicted.to_numpy()
)
results.to_csv(OUTPUT_PATH, index=False)

print("=" * 55)
print("GRIDCASTAI — SEASONAL NAIVE DEMAND BASELINE")
print("=" * 55)
print(f"Total observations: {n}")
print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(val)}")
print(f"Test observations: {len(test)}")
print(f"Evaluated test predictions: {len(results)}")
print("\nTEST METRICS")
print(f"MAE:  {mae:,.2f} MW")
print(f"RMSE: {rmse:,.2f} MW")
print(f"MAPE: {mape:.2f}%")
print(f"\nPredictions saved to: {OUTPUT_PATH}")
print("\nFirst 10 test predictions:")
print(results.head(10).round(2).to_string(index=False))
