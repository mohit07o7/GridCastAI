"""
wind_forecast.py
================
24-hour ahead wind generation forecasting for GridCastAI.

Approach
--------
- Target:   Wind (MW), 24 hours ahead (lag-based, no weather data)
- Features: richer lag set + rolling statistics + calendar signals
            giving XGBoost concrete information that the seasonal-naive
            baseline (lag-24) already encodes — plus week-ago context
            (lag-168), trend proxies (lag-diff-24), and volatility
            measures (rolling std).
- Split:    chronological  70% train / 15% validation / 15% test
- Tuning:   early stopping on validation MAE; no test-set leakage
- Output:   metrics CSV + predictions CSV saved to reports/experiments/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
OUTPUT_DIR = ROOT / "reports" / "experiments"
METRICS_PATH = OUTPUT_DIR / "wind_forecast_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "wind_forecast_predictions.csv"

HORIZON = 24
TARGET_COL = "Wind (MW)"
MODEL_NAME = "XGBoost-Wind-v2"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-and-target DataFrame; rows with any NaN are dropped."""
    series = pd.to_numeric(df[TARGET_COL], errors="raise")
    ts = df["Timestamp"]
    feat = pd.DataFrame(index=df.index)

    # --- Autoregressive lags -------------------------------------------
    # lag_24  ==  the seasonal-naive baseline prediction; XGBoost must
    # learn to beat it, not just discover it.
    for lag in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]:
        feat[f"lag_{lag}"] = series.shift(lag)

    # --- Lag differences (trend / momentum) ---------------------------
    for lag_pair in [(24, 48), (24, 168), (48, 168)]:
        a, b = lag_pair
        feat[f"diff_{a}v{b}"] = series.shift(a) - series.shift(b)

    # --- Rolling statistics (computed on past values only) ------------
    # shift(1) ensures no current-hour leakage.
    past = series.shift(1)
    for window in [6, 12, 24, 48, 72, 168]:
        feat[f"roll_mean_{window}"] = past.rolling(window).mean()
        feat[f"roll_std_{window}"] = past.rolling(window).std()

    # 24-hour min/max captures intra-day range.
    feat["roll_min_24"] = past.rolling(24).min()
    feat["roll_max_24"] = past.rolling(24).max()

    # --- Calendar features -------------------------------------------
    feat["hour"] = ts.dt.hour
    feat["day_of_week"] = ts.dt.dayofweek
    feat["month"] = ts.dt.month
    feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    feat["day_of_year"] = ts.dt.dayofyear

    # Cyclical encoding avoids ordinal distance artifacts.
    feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    # --- Target -------------------------------------------------------
    feat["Actual"] = series.shift(-HORIZON)

    # Bookkeeping columns for chronological splitting.
    feat["_position"] = np.arange(len(df))
    feat["_target_ts"] = ts.shift(-HORIZON)

    return feat.dropna().copy()


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = np.abs(actual) > 1e-8
    mape = (
        float(
            np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero]))
            * 100
        )
        if nonzero.any()
        else float("nan")
    )
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "MAPE (%)": mape}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in source data.")

    n = len(df)
    train_boundary = int(n * 0.70)
    val_boundary = int(n * 0.85)

    print("=" * 70)
    print("GRIDCASTAI — WIND 24-HOUR FORECAST  (XGBoost v2)")
    print("=" * 70)
    print(f"  Total rows:  {n:,}")
    print(f"  Train rows:  {train_boundary:,}  (0-{train_boundary-1})")
    print(f"  Val rows:    {val_boundary - train_boundary:,}  "
          f"({train_boundary}-{val_boundary-1})")
    print(f"  Test rows:   {n - val_boundary:,}  ({val_boundary}-{n-1})")

    data = build_features(df)
    feature_cols = [
        c for c in data.columns
        if c not in ("Actual", "_position", "_target_ts")
    ]

    train = data[data["_position"] < train_boundary]
    val = data[
        (data["_position"] >= train_boundary)
        & (data["_position"] < val_boundary)
    ]
    test = data[data["_position"] >= val_boundary]

    print(f"\n  Feature rows after dropna:")
    print(f"    Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Number of features: {len(feature_cols)}")

    # ---- Seasonal-naive baseline (lag_24 == same-hour yesterday) -------
    naive_actual = test["Actual"].to_numpy()
    naive_pred = test["lag_24"].to_numpy()
    naive_metrics = compute_metrics(naive_actual, naive_pred)

    print(f"\n  Seasonal-naive MAE:  {naive_metrics['MAE (MW)']:.1f} MW")

    # ---- XGBoost model -------------------------------------------------
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:absoluteerror",
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[feature_cols],
        train["Actual"],
        eval_set=[(val[feature_cols], val["Actual"])],
        verbose=False,
    )

    xgb_pred = model.predict(test[feature_cols])
    xgb_metrics = compute_metrics(naive_actual, xgb_pred)

    print(f"  XGBoost MAE:         {xgb_metrics['MAE (MW)']:.1f} MW")
    print(f"  Best iteration:      {model.best_iteration}")
    delta = xgb_metrics["MAE (MW)"] - naive_metrics["MAE (MW)"]
    direction = "BETTER" if delta < 0 else "WORSE"
    print(f"  vs naive:            {delta:+.1f} MW  ({direction})")

    # ---- Feature importance (top 15) -----------------------------------
    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\n  Top-15 feature importances:")
    for feat_name, score in importance.items():
        print(f"    {feat_name:<22}  {score:.4f}")

    # ---- Save outputs --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for model_label, pred, met in [
        ("Seasonal-naive", naive_pred, naive_metrics),
        (MODEL_NAME, xgb_pred, xgb_metrics),
    ]:
        metrics_rows.append({
            "Target": TARGET_COL,
            "Model": model_label,
            "Forecast Horizon (hours)": HORIZON,
            "Train Observations": len(train),
            "Validation Observations": len(val),
            "Test Observations": len(test),
            "Test Start": test["_target_ts"].min(),
            "Test End": test["_target_ts"].max(),
            "Best Iteration": (
                model.best_iteration
                if model_label == MODEL_NAME
                else None
            ),
            **met,
        })

    pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

    pd.DataFrame({
        "Target Timestamp": test["_target_ts"].to_numpy(),
        "Actual (MW)": naive_actual,
        "Seasonal-naive Prediction (MW)": naive_pred,
        "XGBoost Prediction (MW)": xgb_pred,
    }).to_csv(PREDICTIONS_PATH, index=False)

    print(f"\n  Metrics     -> {METRICS_PATH}")
    print(f"  Predictions -> {PREDICTIONS_PATH}")
    print("  Raw data was not modified.")


if __name__ == "__main__":
    main()
