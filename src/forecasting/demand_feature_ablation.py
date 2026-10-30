
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.features import build_features, DEMAND_FEATURES, RENEWABLE_FEATURES
from common.splits import chronological_split


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"

METRICS_PATH = OUTPUT_DIR / "demand_feature_ablation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "demand_feature_ablation_predictions.csv"

# ---------------------------------------------------------
# 1. Load and validate the data
# ---------------------------------------------------------
df = build_features(pd.read_csv(DATA_PATH, parse_dates=['Timestamp']))
demand_features = DEMAND_FEATURES
renewable_features = RENEWABLE_FEATURES
all_features = demand_features + renewable_features

model_df = df.dropna(
    subset=all_features + ["Target"]
).reset_index(drop=True)

if len(model_df) < 100:
    raise ValueError(f"Unexpectedly few usable rows: {len(model_df)}")

# ---------------------------------------------------------
# 3. Identical chronological train/validation/test split
# ---------------------------------------------------------
n = len(model_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train, validation, test = chronological_split(model_df)

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("One or more data splits are empty.")

y_train = train["Target"]
y_val = validation["Target"]
y_test = test["Target"]

# Match the settings in xgboost_demand_forecast.py.
def create_model():
    return XGBRegressor(
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


def evaluate(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return {
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "MAPE (%)": mape,
    }


# ---------------------------------------------------------
# 4. Train demand-only and demand-plus-renewables models
# ---------------------------------------------------------
experiments = [
    ("Demand-only", demand_features),
    ("Demand + renewables", all_features),
]

metric_rows = []
prediction_frames = []

print("=" * 68)
print("GRIDCASTAI — DEMAND FEATURE ABLATION")
print("=" * 68)
print(f"Usable rows: {n:,}")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows: {len(test):,}")
print(f"Test target period: "
      f"{(test['Timestamp'] + pd.Timedelta(hours=24)).min()} "
      f"to "
      f"{(test['Timestamp'] + pd.Timedelta(hours=24)).max()}")

for experiment_name, features in experiments:
    model = create_model()

    model.fit(
        train[features],
        y_train,
        eval_set=[(validation[features], y_val)],
        verbose=False,
    )

    predictions = model.predict(test[features])
    scores = evaluate(y_test, predictions)

    metric_rows.append({
        "Model": experiment_name,
        "Feature Count": len(features),
        "Best Iteration": model.best_iteration,
        "Train Rows": len(train),
        "Validation Rows": len(validation),
        "Test Rows": len(test),
        "Test Start": (
            test["Timestamp"].min() + pd.Timedelta(hours=24)
        ),
        "Test End": (
            test["Timestamp"].max() + pd.Timedelta(hours=24)
        ),
        **scores,
    })

    prediction_frames.append(pd.DataFrame({
        "Forecast Origin": test["Timestamp"].to_numpy(),
        "Target Timestamp": (
            test["Timestamp"] + pd.Timedelta(hours=24)
        ).to_numpy(),
        "Actual Demand (MW)": y_test.to_numpy(),
        "Model": experiment_name,
        "Predicted Demand (MW)": predictions,
        "Absolute Error (MW)": np.abs(
            y_test.to_numpy() - predictions
        ),
    }))

# ---------------------------------------------------------
# 5. Compare performance and save results
# ---------------------------------------------------------
metrics = pd.DataFrame(metric_rows)

demand_mae = metrics.loc[
    metrics["Model"] == "Demand-only", "MAE (MW)"
].iloc[0]

combined_mae = metrics.loc[
    metrics["Model"] == "Demand + renewables", "MAE (MW)"
].iloc[0]

demand_rmse = metrics.loc[
    metrics["Model"] == "Demand-only", "RMSE (MW)"
].iloc[0]

combined_rmse = metrics.loc[
    metrics["Model"] == "Demand + renewables", "RMSE (MW)"
].iloc[0]

mae_improvement = (
    (demand_mae - combined_mae) / demand_mae
) * 100

rmse_improvement = (
    (demand_rmse - combined_rmse) / demand_rmse
) * 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metrics.to_csv(METRICS_PATH, index=False)
pd.concat(prediction_frames, ignore_index=True).to_csv(
    PREDICTIONS_PATH, index=False
)

print("\nTEST RESULTS")
print(metrics[
    ["Model", "Feature Count", "Best Iteration",
     "MAE (MW)", "RMSE (MW)", "MAPE (%)"]
].round(3).to_string(index=False))

print("\nINCREMENTAL VALUE OF RENEWABLE FEATURES")
print(f"MAE improvement:  {mae_improvement:+.2f}%")
print(f"RMSE improvement: {rmse_improvement:+.2f}%")

if mae_improvement > 0 and rmse_improvement > 0:
    print("Interpretation: renewable features improved both metrics.")
elif mae_improvement < 0 and rmse_improvement < 0:
    print("Interpretation: renewable features worsened both metrics.")
else:
    print("Interpretation: results are mixed; inspect both metrics.")

print(f"\nMetrics saved to: {METRICS_PATH}")
print(f"Predictions saved to: {PREDICTIONS_PATH}")
