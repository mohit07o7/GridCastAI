"""
solar_forecast.py
=================
24-hour ahead solar generation forecasting for GridCastAI.

Solar is structurally different from wind and demand:
- Generation is zero at night (hours 21-05 approximately).
- The day/night transition creates sharp discontinuities.
- The seasonal-naive baseline (same hour yesterday) is strong precisely
  because solar follows a near-deterministic daily cycle.

Solar-specific design choices
------------------------------
1. is_daylight flag  — lets XGBoost learn immediately whether the
   target hour will be night (prediction should be near zero).
2. Predicted values clipped to >= 0  — solar generation cannot
   be negative; clipping is a physical constraint, not a model fix.
3. Day-of-year and solar-angle proxy (sin of day-of-year)  — captures
   the seasonal arc of sunrise/sunset times and peak insolation.
4. Same rich lag / rolling / cyclical features used for wind.

Split:  chronological  70% train / 15% validation / 15% test
Tuning: early stopping on validation MAE; no test-set leakage.
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
METRICS_PATH = OUTPUT_DIR / "solar_forecast_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "solar_forecast_predictions.csv"

HORIZON = 24
TARGET_COL = "Solar (MW) corrected"
MODEL_NAME = "XGBoost-Solar-v2"

# Hours (of the TARGET timestamp) expected to have zero solar output.
# Clipping is applied after prediction based on this heuristic.
NIGHT_HOURS = set(range(0, 6)) | set(range(21, 24))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the solar feature matrix.
    The target hour is HORIZON hours after the forecast origin.
    All features are computed from information available at the origin.
    """
    series = pd.to_numeric(df[TARGET_COL], errors="raise")
    ts = df["Timestamp"]
    feat = pd.DataFrame(index=df.index)

    # --- Autoregressive lags -------------------------------------------
    # Note: lag_8760 (year-ago same hour) would address seasonal distribution
    # shift for solar but requires >= 8,760 training rows of history before the
    # first lag-complete row, which wipes out most of our training data given
    # only 18 months of observations. It would be viable with >= 2 full years.
    # lag_336 (2 weeks ago same hour) is the longest feasible periodic lag
    # here; combined with a 2-week rolling mean it anchors the model to recent
    # irradiance levels without excessive data loss.
    for lag in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 336]:
        feat[f"lag_{lag}"] = series.shift(lag)

    # --- Lag differences (trend / momentum) ---------------------------
    for a, b in [(24, 48), (24, 168), (48, 168)]:
        feat[f"diff_{a}v{b}"] = series.shift(a) - series.shift(b)

    # --- Rolling statistics -------------------------------------------
    past = series.shift(1)
    for window in [6, 12, 24, 48, 72, 168, 336]:
        feat[f"roll_mean_{window}"] = past.rolling(window).mean()
        feat[f"roll_std_{window}"] = past.rolling(window).std()

    feat["roll_min_24"] = past.rolling(24).min()
    feat["roll_max_24"] = past.rolling(24).max()

    # --- Calendar features for the FORECAST ORIGIN hour ---------------
    feat["hour"] = ts.dt.hour
    feat["day_of_week"] = ts.dt.dayofweek
    feat["month"] = ts.dt.month
    feat["year"] = ts.dt.year          # captures year-on-year capacity growth
    feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    feat["day_of_year"] = ts.dt.dayofyear

    # Cyclical encoding.
    feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    # Solar angle proxy: day-of-year sine captures insolation seasonality.
    feat["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    feat["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)

    # --- Target-hour features -----------------------------------------
    # The model is told at training time what hour it is forecasting.
    target_ts = ts + pd.Timedelta(hours=HORIZON)
    feat["target_hour"] = target_ts.dt.hour
    feat["target_hour_sin"] = np.sin(2 * np.pi * target_ts.dt.hour / 24)
    feat["target_hour_cos"] = np.cos(2 * np.pi * target_ts.dt.hour / 24)
    feat["target_doy_sin"] = np.sin(2 * np.pi * target_ts.dt.dayofyear / 365)
    feat["target_doy_cos"] = np.cos(2 * np.pi * target_ts.dt.dayofyear / 365)

    # Binary flag: is the TARGET hour a night hour?
    feat["target_is_night"] = target_ts.dt.hour.isin(NIGHT_HOURS).astype(int)

    # --- Label --------------------------------------------------------
    feat["Actual"] = series.shift(-HORIZON)

    # Bookkeeping.
    feat["_position"] = np.arange(len(df))
    feat["_target_ts"] = target_ts

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
    print("GRIDCASTAI — SOLAR 24-HOUR FORECAST  (XGBoost v2)")
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

    # ---- Seasonal-naive baseline ----------------------------------------
    naive_actual = test["Actual"].to_numpy()
    naive_pred = np.clip(test["lag_24"].to_numpy(), 0, None)
    naive_metrics = compute_metrics(naive_actual, naive_pred)

    print(f"\n  Seasonal-naive MAE (clipped):  {naive_metrics['MAE (MW)']:.1f} MW")

    # ---- XGBoost model -------------------------------------------------
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
        train[feature_cols],
        train["Actual"],
        eval_set=[(val[feature_cols], val["Actual"])],
        verbose=False,
    )

    # Clip negative predictions to zero — solar cannot be negative.
    xgb_pred_raw = model.predict(test[feature_cols])
    xgb_pred = np.clip(xgb_pred_raw, 0, None)

    xgb_metrics = compute_metrics(naive_actual, xgb_pred)
    xgb_raw_metrics = compute_metrics(naive_actual, xgb_pred_raw)

    negative_count = int((xgb_pred_raw < 0).sum())
    night_mask = test["target_is_night"].to_numpy().astype(bool)
    neg_pct_night = (
        (xgb_pred_raw[night_mask] < 0).mean() * 100 if night_mask.any() else 0
    )

    print(f"  XGBoost MAE (raw):             {xgb_raw_metrics['MAE (MW)']:.1f} MW")
    print(f"  XGBoost MAE (clipped):         {xgb_metrics['MAE (MW)']:.1f} MW")
    print(f"  Best iteration:                {model.best_iteration}")
    print(f"  Negative predictions (raw):    {negative_count:,}")
    print(f"  Neg predictions in night hrs:  {neg_pct_night:.1f}%")
    delta = xgb_metrics["MAE (MW)"] - naive_metrics["MAE (MW)"]
    direction = "BETTER" if delta < 0 else "WORSE"
    print(f"  vs naive (clipped):            {delta:+.1f} MW  ({direction})")

    # ---- Feature importance (top 15) -----------------------------------
    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\n  Top-15 feature importances:")
    for feat_name, score in importance.items():
        print(f"    {feat_name:<26}  {score:.4f}")

    # ---- Save outputs --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for model_label, pred, met in [
        ("Seasonal-naive (clipped)", naive_pred, naive_metrics),
        (MODEL_NAME + " (raw)", xgb_pred_raw, xgb_raw_metrics),
        (MODEL_NAME + " (clipped)", xgb_pred, xgb_metrics),
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
                if MODEL_NAME in model_label
                else None
            ),
            **met,
        })

    pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

    pd.DataFrame({
        "Target Timestamp": test["_target_ts"].to_numpy(),
        "Actual (MW)": naive_actual,
        "Seasonal-naive Prediction (MW)": naive_pred,
        "XGBoost Raw Prediction (MW)": xgb_pred_raw,
        "XGBoost Clipped Prediction (MW)": xgb_pred,
    }).to_csv(PREDICTIONS_PATH, index=False)

    print(f"\n  Metrics     -> {METRICS_PATH}")
    print(f"  Predictions -> {PREDICTIONS_PATH}")
    print("  Raw data was not modified.")


if __name__ == "__main__":
    main()
