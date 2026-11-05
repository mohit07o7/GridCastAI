
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.features import build_features, DEMAND_FEATURES, RENEWABLE_FEATURES
from common.splits import chronological_split
from common.demand_model import save_demand_bundle, sha256


import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
PREDICTION_PATH = PROJECT_ROOT / "reports" / "experiments" / "xgboost_demand_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "experiments" / "xgboost_demand_metrics.csv"

# 1. Load and validate the hourly demand data.
# Source: demand_model_data.csv built from full_hourly_stacked.csv
#         (Sep 2021 – Jun 2025, 33 k rows, demand + wind + solar)
df = build_features(pd.read_csv(DATA_PATH, parse_dates=['Timestamp']))
feature_cols = ['demand_lag_1', 'demand_lag_24', 'demand_lag_48', 'demand_lag_168', 'demand_mean_24h', 'demand_mean_168h', 'wind_lag_1', 'wind_lag_24', 'solar_lag_1', 'solar_lag_24', 'hour', 'day_of_week', 'month']

# 4. Remove rows only where feature construction is impossible.
# The original source files remain unchanged.
model_df = df.dropna(
    subset=feature_cols + ["Target"]
).reset_index(drop=True)

n = len(model_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Purge 24 rows at each boundary for the 24-hour-ahead target.
purge = 24

train, validation, test = chronological_split(model_df)

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

# 6. Evaluate the historical eligible test segment.
predictions = model.predict(X_test)

# Check against the pre-existing experiment before writing any refreshed outputs.
reference_comparison = {'existing_csv_present': PREDICTION_PATH.exists()}
if PREDICTION_PATH.exists():
    previous = pd.read_csv(PREDICTION_PATH)
    if len(previous) != len(test):
        raise ValueError('Existing test row count changed; investigate before replacing results')
    for column, values in [('Forecast Origin', test.Timestamp),
                           ('Target Timestamp', test['Target Timestamp'])]:
        if not pd.to_datetime(previous[column]).reset_index(drop=True).equals(values.reset_index(drop=True)):
            raise ValueError(f'Existing {column} alignment changed')
    np.testing.assert_allclose(previous['Actual Demand (MW)'], y_test, rtol=0, atol=1e-9)
    np.testing.assert_allclose(previous['XGBoost Forecast (MW)'], predictions, rtol=0, atol=0.02)
    reference_comparison.update({'sha256': sha256(PREDICTION_PATH),
        'max_prediction_difference_mw': float(np.max(np.abs(previous['XGBoost Forecast (MW)'] - predictions))),
        'rows': len(previous)})

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


# Persist this exact fitted model; do not refit on validation or test observations.
metadata = save_demand_bundle(
    model, PROJECT_ROOT / 'models/demand_v1', feature_cols,
    {'train': train, 'validation': validation, 'test': test}, DATA_PATH,
    [Path(__file__), PROJECT_ROOT / 'src/common/features.py',
     PROJECT_ROOT / 'src/common/splits.py', PROJECT_ROOT / 'src/common/demand_model.py'],
    PROJECT_ROOT, {'MAE (MW)': float(mae), 'RMSE (MW)': float(rmse), 'MAPE (%)': float(mape)},
)
# Float64 serialization preserves the exact float32 predictions for load-only replay.
reference = output.copy()
reference['XGBoost Forecast (MW)'] = predictions.astype(np.float64)
reference.to_csv(PROJECT_ROOT / 'models/demand_v1/test_reference.csv', index=False)
import json
(PROJECT_ROOT / 'models/demand_v1/training_reference.json').write_text(json.dumps({
    'model_sha256': metadata['model_sha256'],
    'test_reference_sha256': sha256(PROJECT_ROOT / 'models/demand_v1/test_reference.csv'),
    'existing_experiment_comparison': reference_comparison,
}, indent=2, allow_nan=False) + '\n')
print('Saved model bundle: models/demand_v1')
