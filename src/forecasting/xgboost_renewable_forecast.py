
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/hourly_grid_data_validated.csv"

METRICS_PATH = ROOT / "reports/experiments/renewable_lag0_metrics.csv"
PREDICTIONS_PATH = ROOT / "reports/experiments/renewable_lag0_predictions.csv"

HORIZON = 24

TARGETS = {
    "Wind (MW)": "Wind (MW)",
    "Solar (MW)": "Solar (MW) corrected",
}


def metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE is unstable near zero, especially for solar generation.
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


def build_dataset(df, source_column):
    series = pd.to_numeric(df[source_column], errors="raise")
    timestamps = df["Timestamp"]

    features = pd.DataFrame(index=df.index)

    # Include the current known reading (lag 0).
    for lag in [0, 1, 2, 3, 24, 48, 72, 168]:
        features[f"lag_{lag}"] = series.shift(lag)

    for window in [6, 24, 48, 168]:
        past = series.shift(1)
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
    result["Position"] = np.arange(len(df))

    return result.dropna().copy()


def main():
    df = pd.read_csv(DATA, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    all_metrics = []
    all_predictions = []

    for label, source_column in TARGETS.items():
        print(f"\nTraining {label}...")

        data = build_dataset(df, source_column)

        train = data[data["Position"] < train_end]

        validation = data[
            (data["Position"] >= train_end)
            & (data["Position"] < validation_end)
        ]

        test = data[data["Position"] >= validation_end]

        feature_columns = [
            c for c in data.columns
            if c not in ["Actual", "Position"]
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

        predicted = model.predict(test[feature_columns])
        actual = test["Actual"].to_numpy()

        mae, rmse, mape = metrics(actual, predicted)

        all_metrics.append({
            "Target": label,
            "Forecast Horizon (hours)": HORIZON,
            "Model": "XGBoost with lag 0",
            "Train Observations": len(train),
            "Validation Observations": len(validation),
            "Test Observations": len(test),
            "Best Iteration": getattr(model, "best_iteration", None),
            "MAE (MW)": mae,
            "RMSE (MW)": rmse,
            "MAPE (%)": mape,
        })

        all_predictions.append(pd.DataFrame({
            "Timestamp": df.loc[
                test["Position"], "Timestamp"
            ].to_numpy(),
            "Target": label,
            "Actual (MW)": actual,
            "Predicted (MW)": predicted,
        }))

        print(f"Test MAE: {mae:.2f} MW")
        print(f"Test RMSE: {rmse:.2f} MW")
        print(f"Test MAPE: {mape:.2f}%")

    # Save results using the variable names defined above.
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_metrics).to_csv(
        METRICS_PATH, index=False
    )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        PREDICTIONS_PATH, index=False
    )

    print("\nRenewable forecasting metrics:")
    print(pd.DataFrame(all_metrics).to_string(index=False))
    print("\nMetrics saved to:", METRICS_PATH)
    print("Predictions saved to:", PREDICTIONS_PATH)
    print("Raw data was not modified.")


if __name__ == "__main__":
    main()
