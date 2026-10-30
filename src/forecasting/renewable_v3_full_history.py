"""Four-year, 24-hour-ahead renewable forecasting with honest baselines.

Run: python src/forecasting/renewable_v3_full_history.py

Forecasts are issued after the observation at origin hour t is available.
The target is t+24; the true prior-day baseline is therefore y(t).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "full_hourly_stacked.csv"
EXPERIMENTS = ROOT / "reports" / "experiments"
MODELS = ROOT / "models"
HORIZON = 24
ANNUAL_HOURS = 8760
LATITUDE_DEG = 23.0
LONGITUDE_DEG = 80.0
LOCAL_UTC_OFFSET_HOURS = 5.5
MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": 4,
    "tree_method": "hist",
}


def solar_geometry(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Approximate solar elevation at an India centroid; a feature, not a MW bound."""
    day = ts.dt.dayofyear.to_numpy(dtype=float)
    clock = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60
    gamma = 2 * np.pi * (day - 1 + (clock - 12) / 24) / 365
    eq_minutes = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    solar_minutes = clock * 60 + eq_minutes + 4 * LONGITUDE_DEG - 60 * LOCAL_UTC_OFFSET_HOURS
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    lat = np.deg2rad(LATITUDE_DEG)
    cos_zenith = (
        np.sin(lat) * np.sin(declination)
        + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )
    elevation = np.rad2deg(np.arcsin(np.clip(cos_zenith, -1, 1)))
    return elevation, np.maximum(cos_zenith, 0)


def frozen_run_mask(values: pd.Series, minimum: int = 6) -> pd.Series:
    run = values.ne(values.shift()).cumsum()
    return run.map(run.value_counts()).ge(minimum)


def build_frame(raw: pd.DataFrame, target: str, bad: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    value = raw[target].where(~bad)
    origin = raw["timestamp"]
    target_ts = origin + pd.Timedelta(hours=HORIZON)
    frame = pd.DataFrame({
        "origin": origin,
        "target_timestamp": target_ts,
        "origin_position": np.arange(len(raw)),
        "actual": value.shift(-HORIZON),
        "prior_day": value,
    })
    for hours in [1, 24, 48, 168]:
        frame[f"origin_lag_{hours}"] = value.shift(hours)
    # Same target hour 365 days earlier. In a leap year this is a fixed 365-day lag.
    frame["target_365d_ago"] = value.shift(ANNUAL_HOURS - HORIZON)
    for window in [24, 168]:
        frame[f"recent_mean_{window}"] = value.rolling(window, min_periods=window).mean()
        frame[f"recent_std_{window}"] = value.rolling(window, min_periods=window).std()
    hour = target_ts.dt.hour
    day = target_ts.dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["target_year"] = target_ts.dt.year
    frame["target_month"] = target_ts.dt.month
    frame["target_is_06_20"] = hour.between(6, 20)
    if target == "solar_mw":
        elevation, clear_sky_shape = solar_geometry(target_ts)
        frame["solar_elevation_proxy_deg"] = elevation
        frame["clear_sky_shape_proxy"] = clear_sky_shape
    features = [
        "prior_day", "origin_lag_1", "origin_lag_24", "origin_lag_48",
        "origin_lag_168", "target_365d_ago", "recent_mean_24", "recent_std_24",
        "recent_mean_168", "recent_std_168", "hour_sin", "hour_cos",
        "day_sin", "day_cos", "target_year", "target_month",
    ]
    if target == "solar_mw":
        features += ["solar_elevation_proxy_deg", "clear_sky_shape_proxy"]
    frame["target_frozen_wind"] = (
        frozen_run_mask(raw["wind_mw"]).shift(-HORIZON).fillna(False).astype(bool)
        if target == "wind_mw" else False
    )
    frame = frame.dropna(subset=features + ["actual"]).copy()
    frame["delta"] = frame["actual"] - frame["prior_day"]
    return frame, features


def score(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = actual - prediction
    return {
        "MAE (MW)": float(np.mean(np.abs(error))),
        "RMSE (MW)": float(np.sqrt(np.mean(error**2))),
        "WAPE (%)": float(100 * np.abs(error).sum() / np.abs(actual).sum()),
        "Mean error (actual-predicted MW)": float(error.mean()),
    }


def daily_bootstrap_improvement(
    frame: pd.DataFrame, baseline: str, candidate: str, repeats: int = 1000
) -> dict:
    """Paired day-block CI for candidate MAE improvement in MW."""
    paired = frame[["target_timestamp", "actual", baseline, candidate]].copy()
    paired["improvement"] = (
        (paired["actual"] - paired[baseline]).abs()
        - (paired["actual"] - paired[candidate]).abs()
    )
    daily = paired.groupby(paired["target_timestamp"].dt.date)["improvement"].agg(["sum", "count"])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(daily), size=(repeats, len(daily)))
    values = daily["sum"].to_numpy()[draws].sum(axis=1) / daily["count"].to_numpy()[draws].sum(axis=1)
    return {
        "mean_improvement_mw": float(paired["improvement"].mean()),
        "ci95_low_mw": float(np.percentile(values, 2.5)),
        "ci95_high_mw": float(np.percentile(values, 97.5)),
        "calendar_day_blocks": len(daily),
    }


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        **MODEL_PARAMS,
    )
    model.fit(
        train[features], train["delta"],
        eval_set=[(validation[features], validation["delta"])], verbose=False,
    )
    return model


def refit_model(pretest: pd.DataFrame, features: list[str], best_iteration: int) -> XGBRegressor:
    model = XGBRegressor(n_estimators=best_iteration + 1, **MODEL_PARAMS)
    model.fit(pretest[features], pretest["delta"], verbose=False)
    return model


def evaluate_rolling(frame: pd.DataFrame, features: list[str], target: str, n: int) -> list[dict]:
    """Three expanding-origin folds entirely before the final validation window."""
    rows = []
    for fold, cut in enumerate([int(n * 0.50), int(n * 0.58), int(n * 0.66)], start=1):
        train = frame[frame["origin_position"] < cut - HORIZON]
        validation = frame[(frame["origin_position"] >= cut)
                           & (frame["origin_position"] < cut + 1200)]
        if target == "wind":
            train = train[~train["target_frozen_wind"]]
        else:
            train = train[train["target_is_06_20"]]
            validation = validation[validation["target_is_06_20"]]
        if len(train) == 0 or len(validation) == 0:
            raise ValueError(f"Empty rolling fold {fold} for {target}")
        model = XGBRegressor(n_estimators=100, **MODEL_PARAMS)
        model.fit(train[features], train["delta"], verbose=False)
        baseline = validation["prior_day"].to_numpy()
        candidate = baseline + model.predict(validation[features])
        if target == "solar":
            candidate = np.clip(candidate, 0, None)
        actual = validation["actual"].to_numpy()
        baseline_mae = score(actual, baseline)["MAE (MW)"]
        model_mae = score(actual, candidate)["MAE (MW)"]
        rows.append({
            "target": target, "fold": fold,
            "train_rows": len(train), "validation_rows": len(validation),
            "validation_start": str(validation["target_timestamp"].min()),
            "validation_end": str(validation["target_timestamp"].max()),
            "baseline_mae_mw": baseline_mae, "residual_xgb_mae_mw": model_mae,
            "xgb_improvement_mw": baseline_mae - model_mae,
        })
    return rows


def predictions(frame: pd.DataFrame, model: XGBRegressor, features: list[str], solar: bool) -> pd.DataFrame:
    result = frame[["origin", "target_timestamp", "actual", "prior_day", "target_is_06_20", "target_frozen_wind"]].copy()
    result["residual_xgb"] = result["prior_day"].to_numpy() + model.predict(frame[features])
    if solar:
        result["residual_xgb"] = result["residual_xgb"].clip(lower=0)
        result["xgb_day_persistence_night"] = np.where(
            result["target_is_06_20"], result["residual_xgb"], result["prior_day"]
        )
        result["xgb_day_zero_night"] = np.where(
            result["target_is_06_20"], result["residual_xgb"], 0.0
        )
        result["baseline_day_zero_night"] = np.where(
            result["target_is_06_20"], result["prior_day"], 0.0
        )
    return result


def report_scores(target: str, split: str, pred: pd.DataFrame, columns: list[str]) -> list[dict]:
    segments = {"all": pred}
    if target == "solar":
        segments["06-20"] = pred[pred["target_is_06_20"]]
        segments["21-05"] = pred[~pred["target_is_06_20"]]
    elif target == "wind":
        segments["non-frozen target"] = pred[~pred["target_frozen_wind"]]
    rows = []
    for segment, group in segments.items():
        for col in columns:
            rows.append({"target": target, "split": split, "segment": segment,
                         "model": col, "observations": len(group),
                         "start": str(group["target_timestamp"].min()),
                         "end": str(group["target_timestamp"].max()),
                         **score(group["actual"].to_numpy(), group[col].to_numpy())})
    return rows


def main() -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw["timestamp"].duplicated().any() or raw["timestamp"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Source timestamps must be unique and hourly")
    all_zero = (raw[["demand_mw", "wind_mw", "solar_mw", "total_gen_mw"]] == 0).all(axis=1)
    component_dropout = (raw["solar_wind_mw"] - raw["solar_mw"] - raw["wind_mw"]).abs() > 1
    bad = all_zero | component_dropout
    solar_flatline = frozen_run_mask(raw["solar_mw"]) & raw["solar_mw"].gt(500)
    n = len(raw)
    train_cut = int(n * 0.70)
    validation_cut = int(n * 0.85)
    outputs = []
    rolling_outputs = []
    monthly_outputs = []
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)), "source_rows": n,
        "forecast_horizon_hours": HORIZON,
        "origin_observation_available": True,
        "train_origin_cut": str(raw.loc[train_cut, "timestamp"]),
        "test_origin_cut": str(raw.loc[validation_cut, "timestamp"]),
        "purge_hours_before_each_next_split": HORIZON,
        "invalid_source_hours": int(bad.sum()),
        "source_zero_hours": int(all_zero.sum()),
        "component_dropout_hours": int(component_dropout.sum()),
        "positive_solar_flatline_hours": int(solar_flatline.sum()),
        "solar_geometry_proxy": {"latitude_deg": LATITUDE_DEG, "longitude_deg": LONGITUDE_DEG,
                                  "utc_offset_hours": LOCAL_UTC_OFFSET_HOURS,
                                  "interpretation": "India centroid shape feature; not a physical national MW bound"},
        "targets": {},
    }
    for target, column in [("wind", "wind_mw"), ("solar", "solar_mw")]:
        target_bad = bad | solar_flatline if target == "solar" else bad
        frame, features = build_frame(raw, column, target_bad)
        rolling = evaluate_rolling(frame, features, target, n)
        rolling_outputs += rolling
        train = frame[frame["origin_position"] < train_cut - HORIZON]
        validation = frame[(frame["origin_position"] >= train_cut)
                           & (frame["origin_position"] < validation_cut - HORIZON)]
        test = frame[frame["origin_position"] >= validation_cut]
        if not (train["target_timestamp"].max() < validation["origin"].min()
                and validation["target_timestamp"].max() < test["origin"].min()):
            raise AssertionError(f"Target availability crosses a split boundary for {target}")
        if target == "wind":
            model_train = train[~train["target_frozen_wind"]]
            model_val = validation[~validation["target_frozen_wind"]]
        else:
            model_train = train[train["target_is_06_20"]]
            model_val = validation[validation["target_is_06_20"]]
        if any(len(x) == 0 for x in [model_train, model_val, test]):
            raise ValueError(f"Empty split for {target}")
        model = fit_model(model_train, model_val, features)
        val_pred = predictions(validation, model, features, target == "solar")
        pretest = pd.concat([model_train, model_val], ignore_index=True)
        final_model = refit_model(pretest, features, model.best_iteration)
        test_pred = predictions(test, final_model, features, target == "solar")
        if target == "wind":
            val_scores = {col: score(val_pred["actual"].to_numpy(), val_pred[col].to_numpy())["MAE (MW)"]
                          for col in ["prior_day", "residual_xgb"]}
            comparison = daily_bootstrap_improvement(val_pred, "prior_day", "residual_xgb")
            rolling_wins = sum(row["xgb_improvement_mw"] > 0 for row in rolling)
            selected = "residual_xgb" if comparison["ci95_low_mw"] > 0 and rolling_wins >= 2 else "prior_day"
            columns = ["prior_day", "residual_xgb", "selected"]
            val_pred["selected"] = val_pred[selected]
            test_pred["selected"] = test_pred[selected]
            selection = {"all_hours": selected, "validation_mae": val_scores,
                         "daily_bootstrap": comparison, "positive_rolling_folds": rolling_wins}
        else:
            day = val_pred[val_pred["target_is_06_20"]]
            night = val_pred[~val_pred["target_is_06_20"]]
            day_scores = {col: score(day["actual"].to_numpy(), day[col].to_numpy())["MAE (MW)"]
                          for col in ["prior_day", "residual_xgb"]}
            night_scores = {col: score(night["actual"].to_numpy(), night[col].to_numpy())["MAE (MW)"]
                            for col in ["prior_day", "baseline_day_zero_night"]}
            day_comparison = daily_bootstrap_improvement(day, "prior_day", "residual_xgb")
            night_comparison = daily_bootstrap_improvement(night, "prior_day", "baseline_day_zero_night")
            rolling_wins = sum(row["xgb_improvement_mw"] > 0 for row in rolling)
            selected_day = "residual_xgb" if day_comparison["ci95_low_mw"] > 0 and rolling_wins >= 2 else "prior_day"
            selected_night = "baseline_day_zero_night" if night_comparison["ci95_low_mw"] > 0 else "prior_day"
            for pred in [val_pred, test_pred]:
                pred["selected"] = np.where(pred["target_is_06_20"],
                                            pred[selected_day], pred[selected_night])
            selection = {"06-20": selected_day, "21-05": selected_night,
                         "validation_day_mae": day_scores, "validation_night_mae": night_scores,
                         "validation_day_bootstrap": day_comparison,
                         "validation_night_bootstrap": night_comparison,
                         "positive_daylight_rolling_folds": rolling_wins}
            columns = ["prior_day", "baseline_day_zero_night", "xgb_day_persistence_night",
                       "xgb_day_zero_night", "selected"]
        outputs += report_scores(target, "validation", val_pred, columns)
        outputs += report_scores(target, "test", test_pred, columns)
        for month, group in test_pred.groupby(test_pred["target_timestamp"].dt.to_period("M")):
            for candidate in ["prior_day", "selected"]:
                monthly_outputs.append({
                    "target": target, "month": str(month), "model": candidate,
                    "observations": len(group),
                    **score(group["actual"].to_numpy(), group[candidate].to_numpy()),
                })
        test_pred.to_csv(EXPERIMENTS / f"renewable_v3_{target}_predictions.csv", index=False)
        final_model.save_model(MODELS / f"renewable_v3_{target}.json")
        metadata["targets"][target] = {
            "feature_columns": features, "train_rows": len(model_train),
            "validation_rows": len(model_val), "final_refit_rows": len(pretest),
            "test_rows": len(test),
            "last_train_target": str(train["target_timestamp"].max()),
            "first_validation_origin": str(validation["origin"].min()),
            "last_validation_target": str(validation["target_timestamp"].max()),
            "first_test_origin": str(test["origin"].min()),
            "best_iteration": int(model.best_iteration), "selection": selection,
            "rolling_fold_improvements_mw": [row["xgb_improvement_mw"] for row in rolling],
            "test_target_start": str(test["target_timestamp"].min()),
            "test_target_end": str(test["target_timestamp"].max()),
        }
    metrics = pd.DataFrame(outputs)
    metrics.to_csv(EXPERIMENTS / "renewable_v3_metrics.csv", index=False)
    pd.DataFrame(rolling_outputs).to_csv(EXPERIMENTS / "renewable_v3_rolling_folds.csv", index=False)
    pd.DataFrame(monthly_outputs).to_csv(EXPERIMENTS / "renewable_v3_monthly_metrics.csv", index=False)
    with (EXPERIMENTS / "renewable_v3_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print("\nTest, all eligible hours:\n", metrics[(metrics["split"] == "test")
                                            & (metrics["segment"] == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
