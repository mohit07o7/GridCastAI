
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
OUTPUT_DIR = ROOT / "reports" / "experiments"

METRICS_PATH = OUTPUT_DIR / "solar_daytime_experiment_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "solar_daytime_experiment_predictions.csv"

HORIZON = 24
TARGET_COL = "Solar (MW) corrected"

# Initial daytime proxy: target hours 06:00 through 18:00 inclusive.
DAY_START = 6
DAY_END = 18


def build_features(df):
    """Build features using information available at the forecast origin."""
    series = pd.to_numeric(df[TARGET_COL], errors="raise")
    ts = df["Timestamp"]

    feat = pd.DataFrame(index=df.index)

    # Historical solar generation.
    for lag in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 336]:
        feat[f"lag_{lag}"] = series.shift(lag)

    # Historical changes.
    for a, b in [(24, 48), (24, 168), (48, 168)]:
        feat[f"diff_{a}v{b}"] = series.shift(a) - series.shift(b)

    # Rolling statistics use only observations before the forecast origin.
    past = series.shift(1)
    for window in [6, 12, 24, 48, 72, 168, 336]:
        feat[f"roll_mean_{window}"] = past.rolling(window).mean()
        feat[f"roll_std_{window}"] = past.rolling(window).std()

    feat["roll_min_24"] = past.rolling(24).min()
    feat["roll_max_24"] = past.rolling(24).max()

    # Origin-time calendar features.
    feat["hour"] = ts.dt.hour
    feat["day_of_week"] = ts.dt.dayofweek
    feat["month"] = ts.dt.month
    feat["year"] = ts.dt.year
    feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    feat["day_of_year"] = ts.dt.dayofyear

    feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    feat["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    feat["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)

    # Target-time features are known at the forecast origin.
    target_ts = ts + pd.Timedelta(hours=HORIZON)
    feat["target_hour"] = target_ts.dt.hour
    feat["target_hour_sin"] = np.sin(2 * np.pi * target_ts.dt.hour / 24)
    feat["target_hour_cos"] = np.cos(2 * np.pi * target_ts.dt.hour / 24)
    feat["target_doy_sin"] = np.sin(2 * np.pi * target_ts.dt.dayofyear / 365)
    feat["target_doy_cos"] = np.cos(2 * np.pi * target_ts.dt.dayofyear / 365)

    # Label and audit fields.
    feat["Actual"] = series.shift(-HORIZON)
    feat["_position"] = np.arange(len(df))
    feat["_target_ts"] = target_ts

    return feat.dropna().copy()


def calculate_metrics(actual, predicted):
    error = predicted - actual
    return {
        "N": len(actual),
        "MAE (MW)": mean_absolute_error(actual, predicted),
        "RMSE (MW)": np.sqrt(mean_squared_error(actual, predicted)),
        "Mean Error (MW)": np.mean(error),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    expected = pd.date_range(
        start=df["Timestamp"].min(),
        end=df["Timestamp"].max(),
        freq="h",
    )
    if not pd.DatetimeIndex(df["Timestamp"]).equals(expected):
        raise ValueError("Source timestamps are not a continuous hourly series.")

    n = len(df)
    train_boundary = int(n * 0.70)
    val_boundary = int(n * 0.85)

    data = build_features(df)

    feature_cols = [
        c for c in data.columns
        if c not in ("Actual", "_position", "_target_ts")
    ]

    # Chronological split by forecast-origin row, matching the existing script.
    train = data[data["_position"] < train_boundary].copy()
    val = data[
        (data["_position"] >= train_boundary)
        & (data["_position"] < val_boundary)
    ].copy()
    test = data[data["_position"] >= val_boundary].copy()

    # Train and validate only on daytime TARGET hours.
    train_day = train[
        train["_target_ts"].dt.hour.between(DAY_START, DAY_END)
    ].copy()
    val_day = val[
        val["_target_ts"].dt.hour.between(DAY_START, DAY_END)
    ].copy()

    # Evaluate only on daytime TARGET hours.
    test_day = test[
        test["_target_ts"].dt.hour.between(DAY_START, DAY_END)
    ].copy()

    if train_day.empty or val_day.empty or test_day.empty:
        raise ValueError("A daytime split is empty; inspect the data and split.")

    print("=" * 72)
    print("GRIDCASTAI — DAYTIME-ONLY SOLAR FORECAST EXPERIMENT")
    print("=" * 72)
    print(f"Source observations: {len(df):,}")
    print(f"Feature rows: {len(data):,}")
    print(f"Train daytime rows: {len(train_day):,}")
    print(f"Validation daytime rows: {len(val_day):,}")
    print(f"Test daytime rows: {len(test_day):,}")
    print(
        "Test target period:",
        test_day["_target_ts"].min(),
        "to",
        test_day["_target_ts"].max(),
    )

    # Seasonal-naïve baseline: previous day's same-hour solar generation.
    naive_pred = np.clip(
      pd.to_numeric(df.loc[test_day["_position"], TARGET_COL]).to_numpy(),
      0,
      None
  )
    actual = test_day["Actual"].to_numpy()

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:absoluteerror",
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_day[feature_cols],
        train_day["Actual"],
        eval_set=[(val_day[feature_cols], val_day["Actual"])],
        verbose=False,
    )

    raw_pred = model.predict(test_day[feature_cols])
    clipped_pred = np.clip(raw_pred, 0, None)

    rows = []
    for name, pred in [
        ("Seasonal-naive", naive_pred),
        ("Daytime XGBoost raw", raw_pred),
        ("Daytime XGBoost clipped", clipped_pred),
    ]:
        result = calculate_metrics(actual, pred)
        result.update({
            "Model": name,
            "Forecast Horizon (hours)": HORIZON,
            "Train Daytime Observations": len(train_day),
            "Validation Daytime Observations": len(val_day),
            "Test Daytime Observations": len(test_day),
            "Test Start": test_day["_target_ts"].min(),
            "Test End": test_day["_target_ts"].max(),
        })
        rows.append(result)

    metrics = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)

    predictions = pd.DataFrame({
        "Target Timestamp": test_day["_target_ts"].to_numpy(),
        "Actual (MW)": actual,
        "Seasonal-naive Prediction (MW)": naive_pred,
        "XGBoost Raw Prediction (MW)": raw_pred,
        "XGBoost Clipped Prediction (MW)": clipped_pred,
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nDAYTIME TEST METRICS")
    print(metrics[["Model", "N", "MAE (MW)", "RMSE (MW)", "Mean Error (MW)"]]
          .round(2).to_string(index=False))

    baseline_mae = metrics.loc[
        metrics["Model"] == "Seasonal-naive", "MAE (MW)"
    ].iloc[0]
    clipped_mae = metrics.loc[
        metrics["Model"] == "Daytime XGBoost clipped", "MAE (MW)"
    ].iloc[0]
    improvement = (baseline_mae - clipped_mae) / baseline_mae * 100

    print(f"\nClipped XGBoost MAE improvement vs baseline: {improvement:.2f}%")
    print("Positive means XGBoost is better; negative means it is worse.")
    print(f"\nMetrics saved: {METRICS_PATH}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print("Original scripts and datasets were not modified.")


if __name__ == "__main__":
    main()
