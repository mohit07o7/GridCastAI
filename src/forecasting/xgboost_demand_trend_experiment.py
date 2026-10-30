
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
PREDICTION_PATH = PROJECT_ROOT / "reports" / "experiments" / "xgboost_demand_trend_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "experiments" / "xgboost_demand_trend_metrics.csv"

# 1. Load and validate the hourly demand data.
# Source: demand_model_data.csv built from full_hourly_stacked.csv
#         (Sep 2021 – Jun 2025, 33 k rows, demand + wind + solar)
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

if df["Timestamp"].diff().dropna().ne(
    pd.Timedelta(hours=1)
).any():
    raise ValueError("Expected continuous hourly observations.")

# 2. Define the target: demand 24 hours in the future.
df["Target"] = df["demand_mw"].shift(-24)

# 3. Build features available at the forecast origin.
for lag in [1, 24, 48, 168]:
    df[f"demand_lag_{lag}"] = df["demand_mw"].shift(lag)

# Shift before rolling so the current hour is not included.
df["demand_mean_24h"] = (
    df["demand_mw"].shift(1).rolling(24).mean()
)
df["demand_mean_168h"] = (
    df["demand_mw"].shift(1).rolling(168).mean()
)
# Recent demand-trend features.
# Positive values indicate rising demand; negative values indicate falling demand.
df["demand_trend_6h"] = (
    df["demand_mw"].shift(1) -
    df["demand_mw"].shift(7)
)

df["demand_trend_24h"] = (
    df["demand_mw"].shift(1) -
    df["demand_mw"].shift(25)
)

# Exogenous regressors: wind and solar lagged by 1h and 24h.
# Lag ensures no look-ahead leakage at forecast time.
for lag in [1, 24]:
    df[f"wind_lag_{lag}"]  = df["wind_mw"].shift(lag)
    df[f"solar_lag_{lag}"] = df["solar_mw"].shift(lag)

df["hour"]        = df["Timestamp"].dt.hour
df["day_of_week"] = df["Timestamp"].dt.dayofweek
df["month"]       = df["Timestamp"].dt.month

feature_cols = [
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_48",
    "demand_lag_168",
    "demand_mean_24h",
    "demand_mean_168h",
    "demand_trend_6h",
    "demand_trend_24h",
    "wind_lag_1",
    "wind_lag_24",
    "solar_lag_1",
    "solar_lag_24",
    "hour",
    "day_of_week",
    "month",
]

# 4. Remove rows only where feature construction is impossible.
# The original source files remain unchanged.
model_df = df.dropna(
    subset=feature_cols + ["Target"]
).reset_index(drop=True)

n = len(model_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = model_df.iloc[:train_end]
validation = model_df.iloc[train_end:val_end]
test = model_df.iloc[val_end:]

X_train = train[feature_cols]
y_train = train["Target"]

X_val = validation[feature_cols]
y_val = validation["Target"]

X_test = test[feature_cols]
y_test = test["Target"]

print("=" * 60)
print("GRIDCASTAI — XGBOOST 24-HOUR-AHEAD DEMAND FORECAST")
print("=" * 60)
print(f"Usable rows: {n}")
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Test rows: {len(test)}")

# 5. Train on training data; use validation for early stopping.
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
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# 6. Evaluate the untouched chronological test segment.
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = (
    np.abs((y_test.to_numpy() - predictions)
           / y_test.to_numpy()).mean() * 100
)

# 7. Seasonal-naive reference for the same target timestamps.
# Target at t+24 is predicted using observed demand at t.
naive_predictions = test["demand_mw"].to_numpy()

naive_mae = mean_absolute_error(y_test, naive_predictions)
naive_rmse = np.sqrt(
    mean_squared_error(y_test, naive_predictions)
)
naive_mape = (
    np.abs((y_test.to_numpy() - naive_predictions)
           / y_test.to_numpy()).mean() * 100
)

PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

output = pd.DataFrame({
    "Forecast Origin": test["Timestamp"],
    "Target Timestamp": test["Timestamp"] + pd.Timedelta(hours=24),
    "Actual Demand (MW)": y_test.to_numpy(),
    "XGBoost Forecast (MW)": predictions,
    "Seasonal Naive Forecast (MW)": naive_predictions,
})
output["XGBoost Absolute Error (MW)"] = np.abs(
    output["Actual Demand (MW)"]
    - output["XGBoost Forecast (MW)"]
)
output.to_csv(PREDICTION_PATH, index=False)

metrics = pd.DataFrame([
    {
        "Model": "XGBoost",
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "MAPE (%)": mape,
    },
    {
        "Model": "Seasonal Naive",
        "MAE (MW)": naive_mae,
        "RMSE (MW)": naive_rmse,
        "MAPE (%)": naive_mape,
    },
])
metrics.to_csv(METRICS_PATH, index=False)

print(f"\nBest boosting iteration: {model.best_iteration}")
print("\nTEST METRICS")
print(metrics.round(2).to_string(index=False))
print(f"\nPredictions saved to: {PREDICTION_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")

print("\nTOP 10 FEATURES")
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols,
).sort_values(ascending=False)
print(importance.head(10).round(4).to_string())
