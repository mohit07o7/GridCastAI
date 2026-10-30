
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/hourly_grid_data_validated.csv"
OUTPUT_DIR = ROOT / "reports/experiments"

HORIZON = 24
TARGETS = {
    "Wind (MW)": "Wind (MW)",
    "Solar (MW)": "Solar (MW) corrected",
}


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

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

    for lag in [0, 1, 2, 3, 24, 48, 72, 168]:
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

    result = features.copy()
    result["Actual"] = series.shift(-HORIZON)
    result["Naive Prediction"] = series
    result["Target Position"] = np.arange(len(df)) + HORIZON
    result["Target Timestamp"] = timestamps.shift(-HORIZON)

    return result.dropna().copy()


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    n = len(df)
    train_boundary = int(n * 0.70)
    validation_boundary = int(n * 0.85)

    all_metrics = []
    all_predictions = []

    for label, column in TARGETS.items():
        print(f"\nTraining {label}...")

        data = build_dataset(df, column)

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
            eval_set=[
                (validation[feature_columns], validation["Actual"])
            ],
            verbose=False,
        )

        actual = test["Actual"].to_numpy()
        xgb_pred = model.predict(test[feature_columns])
        naive_pred = test["Naive Prediction"].to_numpy()

        for model_name, prediction in [
            ("Seasonal-naive", naive_pred),
            ("XGBoost with lag 0", xgb_pred),
        ]:
            mae, rmse, mape = calculate_metrics(actual, prediction)

            all_metrics.append({
                "Target": label,
                "Model": model_name,
                "Forecast Horizon (hours)": HORIZON,
                "Train Observations": len(train),
                "Validation Observations": len(validation),
                "Test Observations": len(test),
                "Test Start": test["Target Timestamp"].min(),
                "Test End": test["Target Timestamp"].max(),
                "Best Iteration": (
                    model.best_iteration
                    if model_name == "XGBoost with lag 0"
                    else np.nan
                ),
                "MAE (MW)": mae,
                "RMSE (MW)": rmse,
                "MAPE (%)": mape,
            })

        all_predictions.append(pd.DataFrame({
            "Target Timestamp": test["Target Timestamp"].to_numpy(),
            "Target": label,
            "Actual (MW)": actual,
            "Seasonal-naive Prediction (MW)": naive_pred,
            "XGBoost Prediction (MW)": xgb_pred,
        }))

        print(f"Training rows: {len(train)}")
        print(f"Validation rows: {len(validation)}")
        print(f"Test rows: {len(test)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "renewable_lag0_target_split_metrics.csv"
    predictions_path = OUTPUT_DIR / "renewable_lag0_target_split_predictions.csv"

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, index=False)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        predictions_path, index=False
    )

    print("\nCorrected comparison:")
    print(metrics_df.to_string(index=False))
    print("\nMetrics saved to:", metrics_path)
    print("Predictions saved to:", predictions_path)
    print("Raw data was not modified.")


if __name__ == "__main__":
    main()
