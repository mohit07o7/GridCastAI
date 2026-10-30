
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/hourly_grid_data_validated.csv"
METRICS_PATH = ROOT / "reports/experiments/renewable_corrected_split_metrics.csv"
PREDICTIONS_PATH = ROOT / "reports/experiments/renewable_corrected_split_predictions.csv"

HORIZON = 24

TARGETS = {
    "Wind (MW)": "Wind (MW)",
    "Solar (MW)": "Solar (MW) corrected",
}


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE is unreliable when actual solar generation is near zero.
    nonzero = np.abs(actual) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return mae, rmse, mape


def build_dataset(df, column):
    series = pd.to_numeric(df[column], errors="raise")
    timestamps = df["Timestamp"]

    features = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 24, 48, 72, 168]:
        features[f"lag_{lag}"] = series.shift(lag)

    past = series.shift(1)
    for window in [6, 24, 48, 168]:
        features[f"rolling_mean_{window}"] = (
            past.rolling(window).mean()
        )
        features[f"rolling_std_{window}"] = (
            past.rolling(window).std()
        )

    features["hour"] = timestamps.dt.hour
    features["day_of_week"] = timestamps.dt.dayofweek
    features["month"] = timestamps.dt.month

    data = features.copy()
    data["Actual"] = series.shift(-HORIZON)
    data["Naive Prediction"] = series
    data["Origin Position"] = np.arange(len(df))
    data["Target Position"] = np.arange(len(df)) + HORIZON
    data["Target Timestamp"] = timestamps.shift(-HORIZON)

    return data.dropna().copy()


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    # These boundaries are defined on TARGET positions.
    n = len(df)
    train_boundary = int(n * 0.70)
    validation_boundary = int(n * 0.85)

    print("Target-timestamp split boundaries:")
    print("Training targets end before:", df.loc[train_boundary, "Timestamp"])
    print("Validation targets end before:", df.loc[validation_boundary, "Timestamp"])

    metric_rows = []
    prediction_frames = []

    for label, column in TARGETS.items():
        print(f"\nTraining {label}...")

        data = build_dataset(df, column)

        # Assign each example according to when its target occurs.
        train = data[data["Target Position"] < train_boundary]
        validation = data[
            (data["Target Position"] >= train_boundary)
            & (data["Target Position"] < validation_boundary)
        ]
        test = data[data["Target Position"] >= validation_boundary]

        feature_columns = [
            c for c in data.columns
            if c not in [
                "Actual",
                "Naive Prediction",
                "Origin Position",
                "Target Position",
                "Target Timestamp",
            ]
        ]

        model = XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train[feature_columns],
            train["Actual"],
            eval_set=[(
                validation[feature_columns],
                validation["Actual"],
            )],
            verbose=False,
        )

        actual = test["Actual"].to_numpy()
        xgb_predictions = model.predict(test[feature_columns])
        naive_predictions = test["Naive Prediction"].to_numpy()

        for model_name, predictions in [
            ("Seasonal-naive", naive_predictions),
            ("XGBoost", xgb_predictions),
        ]:
            mae, rmse, mape = calculate_metrics(actual, predictions)

            metric_rows.append({
                "Target": label,
                "Model": model_name,
                "Forecast Horizon (hours)": HORIZON,
                "Train Observations": len(train),
                "Validation Observations": len(validation),
                "Test Observations": len(test),
                "Test Target Start": test["Target Timestamp"].min(),
                "Test Target End": test["Target Timestamp"].max(),
                "Best Iteration": (
                    getattr(model, "best_iteration", None)
                    if model_name == "XGBoost"
                    else np.nan
                ),
                "MAE (MW)": mae,
                "RMSE (MW)": rmse,
                "MAPE (%)": mape,
            })

        prediction_frames.append(pd.DataFrame({
            "Target Timestamp": test["Target Timestamp"].to_numpy(),
            "Target": label,
            "Actual (MW)": actual,
            "Seasonal-naive Prediction (MW)": naive_predictions,
            "XGBoost Prediction (MW)": xgb_predictions,
        }))

        print(f"Training rows: {len(train)}")
        print(f"Validation rows: {len(validation)}")
        print(f"Test rows: {len(test)}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nCorrected-split metrics:")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nMetrics saved to:", METRICS_PATH)
    print("Predictions saved to:", PREDICTIONS_PATH)
    print("Raw data was not modified.")


if __name__ == "__main__":
    main()
