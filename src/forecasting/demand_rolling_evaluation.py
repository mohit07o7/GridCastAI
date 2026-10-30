
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

FOLD_METRICS_PATH = OUTPUT_DIR / "demand_rolling_fold_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "demand_rolling_summary.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "demand_rolling_predictions.csv"

# Seven test windows, each 5% of the usable observations.
N_FOLDS = 7
TEST_FRACTION = 0.05
VALIDATION_FRACTION = 0.15


def create_model():
    """Use the same hyperparameters as the original demand model."""
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


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE (MW)": mean_absolute_error(actual, predicted),
        "RMSE (MW)": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAPE (%)": np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100,
    }


# ---------------------------------------------------------
# 1. Load data and construct leakage-safe features
# ---------------------------------------------------------
df = build_features(pd.read_csv(DATA_PATH, parse_dates=['Timestamp']))
demand_features = DEMAND_FEATURES
renewable_features = RENEWABLE_FEATURES
all_features = demand_features + renewable_features

model_df = df.dropna(
    subset=all_features + ["Target", "Target Timestamp"]
).reset_index(drop=True)

# Data are continuous and sorted, so target timestamps are sorted too.
n = len(model_df)
test_size = int(n * TEST_FRACTION)
validation_size = int(n * VALIDATION_FRACTION)

if test_size < 1 or validation_size < 1:
    raise ValueError("Not enough observations for the requested splits.")

first_test_start = n - N_FOLDS * test_size

if first_test_start - validation_size <= 0:
    raise ValueError("Not enough history for all requested folds.")

print("=" * 72)
print("GRIDCASTAI — EXPANDING-WINDOW DEMAND EVALUATION")
print("=" * 72)
print(f"Usable observations: {n:,}")
print(f"Validation observations per fold: {validation_size:,}")
print(f"Test observations per fold: {test_size:,}")
print(f"Number of folds: {N_FOLDS}")
print("Split basis: target timestamp")
print("Models: demand-only vs demand + renewables")

fold_rows = []
prediction_frames = []

# ---------------------------------------------------------
# 2. Run seven chronological folds
# ---------------------------------------------------------
for fold in range(N_FOLDS):
    test_start = first_test_start + fold * test_size
    test_end = min(test_start + test_size, n)
    validation_start = test_start - validation_size

    train, validation, test = chronological_split(model_df, validation_start, test_start, test_end)

    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Fold {fold + 1} has an empty split.")

    # The target-time intervals are disjoint and chronological.
    assert (
        train["Target Timestamp"].max()
        < validation["Target Timestamp"].min()
    )
    assert (
        validation["Target Timestamp"].max()
        < test["Target Timestamp"].min()
    )

    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    print(
        f"Train targets: {train['Target Timestamp'].min()} "
        f"to {train['Target Timestamp'].max()}"
    )
    print(
        f"Validation targets: {validation['Target Timestamp'].min()} "
        f"to {validation['Target Timestamp'].max()}"
    )
    print(
        f"Test targets: {test['Target Timestamp'].min()} "
        f"to {test['Target Timestamp'].max()}"
    )
    print(
        f"Rows — train: {len(train):,}, "
        f"validation: {len(validation):,}, test: {len(test):,}"
    )

    for model_name, features in [
        ("Demand-only", demand_features),
        ("Demand + renewables", all_features),
    ]:
        model = create_model()

        model.fit(
            train[features],
            train["Target"],
            eval_set=[
                (validation[features], validation["Target"])
            ],
            verbose=False,
        )

        predictions = model.predict(test[features])
        actual = test["Target"].to_numpy()
        scores = calculate_metrics(actual, predictions)

        fold_rows.append({
            "Fold": fold + 1,
            "Model": model_name,
            "Feature Count": len(features),
            "Train Rows": len(train),
            "Validation Rows": len(validation),
            "Test Rows": len(test),
            "Test Start": test["Target Timestamp"].min(),
            "Test End": test["Target Timestamp"].max(),
            "Best Iteration": model.best_iteration,
            **scores,
        })

        prediction_frames.append(pd.DataFrame({
            "Fold": fold + 1,
            "Model": model_name,
            "Forecast Origin": test["Timestamp"].to_numpy(),
            "Target Timestamp": test["Target Timestamp"].to_numpy(),
            "Actual Demand (MW)": actual,
            "Predicted Demand (MW)": predictions,
            "Absolute Error (MW)": np.abs(actual - predictions),
        }))

        print(
            f"{model_name}: "
            f"MAE={scores['MAE (MW)']:,.2f} MW, "
            f"RMSE={scores['RMSE (MW)']:,.2f} MW, "
            f"MAPE={scores['MAPE (%)']:.3f}%, "
            f"best iteration={model.best_iteration}"
        )

# ---------------------------------------------------------
# 3. Save fold-level metrics and predictions
# ---------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fold_metrics = pd.DataFrame(fold_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)

fold_metrics.to_csv(FOLD_METRICS_PATH, index=False)
predictions_df.to_csv(PREDICTIONS_PATH, index=False)

# Mean and standard deviation describe variation across windows.
summary = (
    fold_metrics.groupby("Model")[
        ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]
    ]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric} — {statistic}"
    for metric, statistic in summary.columns
]
summary = summary.reset_index()

summary.to_csv(SUMMARY_PATH, index=False)

print("\n" + "=" * 72)
print("ROLLING EVALUATION SUMMARY")
print("=" * 72)
print(summary.round(3).to_string(index=False))

# Calculate relative change in the mean fold-level scores.
means = fold_metrics.groupby("Model")[
    ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]
].mean()

print("\nDEMAND + RENEWABLES RELATIVE TO DEMAND-ONLY")
for metric in ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]:
    demand_score = means.loc["Demand-only", metric]
    combined_score = means.loc["Demand + renewables", metric]
    improvement = (demand_score - combined_score) / demand_score * 100

    print(f"{metric}: {improvement:+.2f}% improvement")

print("\nSaved files:")
print(FOLD_METRICS_PATH)
print(SUMMARY_PATH)
print(PREDICTIONS_PATH)
