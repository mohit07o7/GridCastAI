import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("reports/experiments/xgboost_demand_predictions.csv")
out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path, parse_dates=["Forecast Origin", "Target Timestamp"])
df = df.sort_values("Target Timestamp")

actual = "Actual Demand (MW)"
xgb = "XGBoost Forecast (MW)"
baseline = "Seasonal Naive Forecast (MW)"

# Chart 1: Actual vs predicted demand across the test period
plt.figure(figsize=(15, 6))
plt.plot(df["Target Timestamp"], df[actual], label="Actual demand")
plt.plot(df["Target Timestamp"], df[xgb], label="XGBoost forecast")
plt.plot(df["Target Timestamp"], df[baseline], label="Seasonal-naive baseline")
plt.title("GridCastAI — 24-Hour-Ahead Demand Forecast")
plt.xlabel("Target timestamp")
plt.ylabel("Electricity demand (MW)")
plt.legend()
plt.tight_layout()
plt.savefig(out / "demand_forecast_test_period.png", dpi=160)
plt.close()

# Chart 2: Actual vs predicted demand for the first 7 days of the test set
sample = df.head(24 * 7)
plt.figure(figsize=(15, 6))
plt.plot(sample["Target Timestamp"], sample[actual], marker=".", label="Actual demand")
plt.plot(sample["Target Timestamp"], sample[xgb], marker=".", label="XGBoost forecast")
plt.plot(sample["Target Timestamp"], sample[baseline], marker=".", label="Seasonal-naive baseline")
plt.title("GridCastAI — First 7 Days of Test Forecasts")
plt.xlabel("Target timestamp")
plt.ylabel("Electricity demand (MW)")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(out / "demand_forecast_first_7_days.png", dpi=160)
plt.close()

# Chart 3: Forecast error distribution
errors = df[xgb] - df[actual]
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.axvline(0, linestyle="--")
plt.title("GridCastAI — Demand Forecast Error Distribution")
plt.xlabel("Forecast minus actual demand (MW)")
plt.ylabel("Number of hourly forecasts")
plt.tight_layout()
plt.savefig(out / "demand_forecast_error_distribution.png", dpi=160)
plt.close()

print("Charts saved:")
for name in [
    "demand_forecast_test_period.png",
    "demand_forecast_first_7_days.png",
    "demand_forecast_error_distribution.png",
]:
    print(out / name)
