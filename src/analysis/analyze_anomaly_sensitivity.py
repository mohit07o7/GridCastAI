
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
ANOMALY_TIME = pd.Timestamp("2025-04-04 16:00:00")

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Reconstruct the same 24-hour-ahead modelling features.
df["Target"] = df["Demand (MW)"].shift(-24)

for lag in [1, 24, 48, 168]:
    df[f"demand_lag_{lag}"] = df["Demand (MW)"].shift(lag)

df["demand_mean_24h"] = (
    df["Demand (MW)"].shift(1).rolling(24).mean()
)
df["demand_mean_168h"] = (
    df["Demand (MW)"].shift(1).rolling(168).mean()
)

df["hour"] = df["Timestamp"].dt.hour
df["day_of_week"] = df["Timestamp"].dt.dayofweek
df["month"] = df["Timestamp"].dt.month

features = [
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_48",
    "demand_lag_168",
    "demand_mean_24h",
    "demand_mean_168h",
    "hour",
    "day_of_week",
    "month",
]

df = df.dropna(subset=features + ["Target"]).reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end].copy()

# Each row's forecast target is 24 hours after its origin.
validation["Target Timestamp"] = (
    validation["Timestamp"] + pd.Timedelta(hours=24)
)

anomaly_mask = validation["Target Timestamp"].eq(ANOMALY_TIME)

print("=" * 60)
print("GRIDCASTAI — DEMAND ANOMALY SENSITIVITY")
print("=" * 60)
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Anomaly target found in validation: {anomaly_mask.any()}")

if anomaly_mask.any():
    print("\nANOMALY TARGET ROW")
    print(
        validation.loc[
            anomaly_mask,
            ["Timestamp", "Target Timestamp", "Target"]
        ].to_string(index=False)
    )
else:
    raise ValueError(
        "Anomaly target is not in validation. "
        "Check the split and timestamp alignment."
    )

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    eval_metric="mae",
    random_state=42,
    n_jobs=4,
)

model.fit(
    train[features],
    train["Target"],
    eval_set=[(validation[features], validation["Target"])],
    verbose=False,
)

predictions = model.predict(validation[features])
actual = validation["Target"].to_numpy()

def calculate_metrics(actual, predicted):
    return {
        "MAE (MW)": mean_absolute_error(actual, predicted),
        "RMSE (MW)": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAPE (%)": np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100,
    }

all_metrics = calculate_metrics(actual, predictions)

keep = ~anomaly_mask.to_numpy()
without_anomaly_metrics = calculate_metrics(
    actual[keep],
    predictions[keep],
)

results = pd.DataFrame([
    {"Evaluation": "All validation observations", **all_metrics},
    {
        "Evaluation": "Excluding anomalous target",
        **without_anomaly_metrics,
    },
])

output_path = PROJECT_ROOT / "reports" / "experiments" / "anomaly_sensitivity_validation.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print(f"\nBest boosting iteration: {model.best_iteration}")
print("\nVALIDATION METRICS")
print(results.round(2).to_string(index=False))

print("\nANOMALY FORECAST ERROR")
anomaly_actual = actual[anomaly_mask.to_numpy()][0]
anomaly_prediction = predictions[anomaly_mask.to_numpy()][0]

print(f"Actual demand: {anomaly_actual:,.2f} MW")
print(f"Predicted demand: {anomaly_prediction:,.2f} MW")
print(
    f"Absolute error: "
    f"{abs(anomaly_actual - anomaly_prediction):,.2f} MW"
)

print(f"\nSaved report: {output_path}")
print("\nNo source observations were modified or deleted.")
