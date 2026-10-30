
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
ANOMALY_TIME = pd.Timestamp("2025-04-04 16:00:00")
WINDOW_HOURS = 168

OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "anomaly_feature_impact_full_window.csv"

# Load data without modifying the source file.
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

original_demand = df["Demand (MW)"].copy()
counterfactual_demand = original_demand.copy()

# Locate the anomaly.
positions = np.flatnonzero(
    df["Timestamp"].eq(ANOMALY_TIME).to_numpy()
)

if len(positions) != 1:
    raise ValueError(
        f"Expected exactly one anomaly row; found {len(positions)}."
    )

i = positions[0]

if i == 0 or i == len(df) - 1:
    raise ValueError("Cannot estimate from adjacent observations.")

previous_value = original_demand.iloc[i - 1]
next_value = original_demand.iloc[i + 1]
estimated_value = (previous_value + next_value) / 2

# Counterfactual copy only. Original observations remain unchanged.
counterfactual_demand.iloc[i] = estimated_value

print("=" * 68)
print("GRIDCASTAI — FULL-WINDOW ANOMALY FEATURE-IMPACT ANALYSIS")
print("=" * 68)
print(f"Anomaly timestamp: {ANOMALY_TIME}")
print(f"Original demand: {original_demand.iloc[i]:,.2f} MW")
print(f"Counterfactual estimate: {estimated_value:,.2f} MW")

# Rebuild features consistently for both scenarios.
def build_features(demand):
    result = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Target": demand.shift(-24),
    })

    for lag in [1, 24, 48, 168]:
        result[f"demand_lag_{lag}"] = demand.shift(lag)

    result["demand_mean_24h"] = (
        demand.shift(1).rolling(24).mean()
    )
    result["demand_mean_168h"] = (
        demand.shift(1).rolling(168).mean()
    )

    result["hour"] = result["Timestamp"].dt.hour
    result["day_of_week"] = result["Timestamp"].dt.dayofweek
    result["month"] = result["Timestamp"].dt.month

    return result


original = build_features(original_demand)
counterfactual = build_features(counterfactual_demand)

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

# Retain identical eligible rows in both scenarios.
valid_indices = original.dropna(
    subset=features + ["Target"]
).index

original = original.loc[valid_indices].reset_index(drop=True)
counterfactual = counterfactual.loc[valid_indices].reset_index(drop=True)

# Preserve the original chronological 70% / 15% / 15% split.
n = len(original)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = original.iloc[:train_end]
validation = original.iloc[train_end:val_end]

# Train exactly one model on original training data.
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

# Select the full seven-day forecast-origin window.
window_start = ANOMALY_TIME
window_end = ANOMALY_TIME + pd.Timedelta(hours=WINDOW_HOURS)

window_mask = (
    (original["Timestamp"] > window_start)
    & (original["Timestamp"] <= window_end)
)

window_original = original.loc[window_mask].copy()
window_counterfactual = counterfactual.loc[window_mask].copy()

if window_original.empty:
    raise ValueError("No eligible forecast origins in the window.")

# Compare the same trained model with both feature sets.
pred_original = model.predict(window_original[features])
pred_counterfactual = model.predict(window_counterfactual[features])

actual = window_original["Target"].to_numpy()

results = pd.DataFrame({
    "Forecast Origin": window_original["Timestamp"].to_numpy(),
    "Target Timestamp": (
        window_original["Timestamp"]
        + pd.Timedelta(hours=24)
    ).to_numpy(),
    "Actual Demand (MW)": actual,
    "Original Prediction (MW)": pred_original,
    "Counterfactual Prediction (MW)": pred_counterfactual,
})

results["Prediction Change (MW)"] = (
    results["Counterfactual Prediction (MW)"]
    - results["Original Prediction (MW)"]
)

results["Original Absolute Error (MW)"] = (
    results["Actual Demand (MW)"]
    - results["Original Prediction (MW)"]
).abs()

results["Counterfactual Absolute Error (MW)"] = (
    results["Actual Demand (MW)"]
    - results["Counterfactual Prediction (MW)"]
).abs()

def calculate_metrics(actual_values, predictions):
    return {
        "MAE (MW)": mean_absolute_error(actual_values, predictions),
        "RMSE (MW)": np.sqrt(
            mean_squared_error(actual_values, predictions)
        ),
        "MAPE (%)": np.mean(
            np.abs((actual_values - predictions) / actual_values)
        ) * 100,
    }


comparison = pd.DataFrame([
    {
        "Scenario": "Original input",
        **calculate_metrics(actual, pred_original),
    },
    {
        "Scenario": "Counterfactual input",
        **calculate_metrics(actual, pred_counterfactual),
    },
])

# Summarize the feature-impact window.
absolute_changes = results["Prediction Change (MW)"].abs()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(f"\nTraining rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Best boosting iteration: {model.best_iteration}")
print(f"Requested impact window: {WINDOW_HOURS} hours")
print(f"Eligible forecast origins: {len(results)}")
print(
    f"First forecast origin: "
    f"{results['Forecast Origin'].min()}"
)
print(
    f"Last forecast origin: "
    f"{results['Forecast Origin'].max()}"
)
print(
    f"Distinct target timestamps: "
    f"{results['Target Timestamp'].nunique()}"
)

print("\nMETRICS ON THE SAME FORECAST ORIGINS")
print(comparison.round(2).to_string(index=False))

print("\nABSOLUTE PREDICTION CHANGE (MW)")
print(absolute_changes.describe().round(2).to_string())

print("\nLARGEST PREDICTION CHANGES")
largest = results.loc[
    absolute_changes.nlargest(min(10, len(results))).index,
    [
        "Forecast Origin",
        "Target Timestamp",
        "Actual Demand (MW)",
        "Original Prediction (MW)",
        "Counterfactual Prediction (MW)",
        "Prediction Change (MW)",
    ],
].copy()

# Round only numeric columns to avoid datetime-format warnings.
numeric_columns = largest.select_dtypes(include="number").columns
largest[numeric_columns] = largest[numeric_columns].round(2)
print(largest.to_string(index=False))

print(f"\nSaved detailed report: {OUTPUT_PATH}")
print("Original dataset was not modified.")
