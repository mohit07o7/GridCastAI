"""Audit the stacked hourly data and recheck renewable forecast baselines.

Run from the project root with:
    python src/auditing/audit_full_model_readiness.py

This reads the existing processed data and saved predictions. It does not train
models or change the source data. Results are written under reports/audits/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "full_hourly_stacked.csv"
VALIDATED = ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
EXPERIMENTS = ROOT / "reports" / "experiments"
OUT = ROOT / "reports" / "audits"
MEASURES = ["demand_mw", "wind_mw", "solar_raw_mw", "solar_mw", "solar_wind_mw", "total_gen_mw"]


def era(interval: int) -> str:
    return {300: "5-minute source", 10: "10-second source", 3600: "hourly source"}.get(
        interval, f"unknown ({interval})"
    )


def metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    error = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    a = actual.to_numpy(dtype=float)
    return {
        "MAE (MW)": float(np.mean(np.abs(error))),
        "RMSE (MW)": float(np.sqrt(np.mean(error**2))),
        "WAPE (%)": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(a))),
        "Mean error (actual-predicted MW)": float(np.mean(error)),
    }


def repeated_runs(df: pd.DataFrame, column: str) -> dict:
    values = df[column]
    run_id = values.ne(values.shift()).cumsum()
    runs = df.groupby(run_id, sort=False).agg(
        start=("timestamp", "first"), end=("timestamp", "last"), length=(column, "size")
    )
    long = runs[runs["length"] >= 6]
    return {
        "runs_at_least_6h": int(len(long)),
        "hours_in_runs_at_least_6h": int(long["length"].sum()),
        "longest_run_hours": int(runs["length"].max()),
        "longest_run_start": str(runs.loc[runs["length"].idxmax(), "start"]),
    }


def recheck_baselines() -> pd.DataFrame:
    source = pd.read_csv(VALIDATED, parse_dates=["Timestamp"])
    source = source.set_index("Timestamp")
    rows = []
    for target, filename, actual_col, xgb_col in [
        ("wind", "wind_forecast_predictions.csv", "Wind (MW)", "XGBoost Prediction (MW)"),
        (
            "solar",
            "solar_forecast_predictions.csv",
            "Solar (MW) corrected",
            "XGBoost Clipped Prediction (MW)",
        ),
    ]:
        pred = pd.read_csv(EXPERIMENTS / filename, parse_dates=["Target Timestamp"])
        target_ts = pred["Target Timestamp"]
        observed = source[actual_col].reindex(target_ts).to_numpy()
        if not np.allclose(observed, pred["Actual (MW)"], atol=0.01, rtol=0):
            raise ValueError(f"{target}: predictions do not match validated actuals")
        two_days_ago = source[actual_col].reindex(
            target_ts - pd.Timedelta(hours=48)
        ).to_numpy()
        if target == "solar":
            two_days_ago = np.clip(two_days_ago, 0, None)
        if not np.allclose(
            two_days_ago, pred["Seasonal-naive Prediction (MW)"], atol=0.01, rtol=0
        ):
            raise ValueError(f"{target}: saved baseline is not the expected 48-hour lag")
        yesterday_ts = target_ts - pd.Timedelta(hours=24)
        yesterday = source[actual_col].reindex(yesterday_ts).to_numpy()
        if np.isnan(yesterday).any():
            raise ValueError(f"{target}: missing prior-day source values")
        if target == "solar":
            yesterday = np.clip(yesterday, 0, None)
        pred["true_yesterday"] = yesterday
        pred["hour"] = target_ts.dt.hour
        groups = {"all": pred}
        if target == "solar":
            groups["06-20"] = pred[pred["hour"].between(6, 20)]
            groups["21-05"] = pred[~pred["hour"].between(6, 20)]
        for segment, group in groups.items():
            for name, column in [
                ("Saved baseline (48h before target)", "Seasonal-naive Prediction (MW)"),
                ("True yesterday (24h before target)", "true_yesterday"),
                ("Saved XGBoost", xgb_col),
            ]:
                rows.append(
                    {
                        "target": target,
                        "segment": segment,
                        "model": name,
                        "observations": len(group),
                        "test_start": str(group["Target Timestamp"].min()),
                        "test_end": str(group["Target Timestamp"].max()),
                        **metrics(group["Actual (MW)"], group[column]),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "renewable_baseline_recheck.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    if df["timestamp"].isna().any():
        raise ValueError("Unparseable timestamps")
    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric = MEASURES + ["source_interval_s", "rows_per_hour", "expected_rows_per_hour"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    intervals = df["timestamp"].diff().dt.total_seconds().dropna()
    residual = df["solar_wind_mw"] - df["solar_mw"] - df["wind_mw"]
    all_zero = df[(df[["demand_mw", "wind_mw", "solar_mw", "total_gen_mw"]] == 0).all(axis=1)]
    all_zero[["timestamp", "source_file", "source_interval_s", "rows_per_hour"]].to_csv(
        OUT / "full_dataset_all_zero_hours.csv", index=False
    )
    residual_issues = df.loc[residual.abs() > 1, [
        "timestamp", "source_file", "solar_wind_mw", "solar_mw", "wind_mw"
    ]].copy()
    residual_issues["combined_minus_components_mw"] = residual.loc[residual_issues.index].values
    residual_issues.to_csv(OUT / "full_dataset_solar_wind_residuals.csv", index=False)
    solar_run = df["solar_mw"].ne(df["solar_mw"].shift()).cumsum()
    solar_flatline = df["solar_mw"].gt(500) & solar_run.map(solar_run.value_counts()).ge(6)
    df.loc[solar_flatline, ["timestamp", "source_file", "solar_mw"]].to_csv(
        OUT / "full_dataset_positive_solar_flatline.csv", index=False
    )
    positive = df["demand_mw"].shift(1) > 0
    demand_change = (df["demand_mw"] / df["demand_mw"].shift(1) - 1).abs()
    big_jumps = df.loc[positive & (demand_change > 0.4), ["timestamp", "demand_mw", "source_file"]].copy()
    big_jumps["absolute_fractional_change"] = demand_change.loc[big_jumps.index].values

    df["source_era"] = df["source_interval_s"].map(lambda x: era(int(x)) if pd.notna(x) else "missing")
    era_rows = []
    for name, group in df.groupby("source_era", sort=False):
        row = {
            "source_era": name,
            "start": str(group["timestamp"].min()),
            "end": str(group["timestamp"].max()),
            "hours": len(group),
            "source_files": int(group["source_file"].nunique()),
            "incomplete_hours": int((group["rows_per_hour"] < group["expected_rows_per_hour"]).sum()),
            "solar_negative_raw_hours": int((group["solar_raw_mw"] < 0).sum()),
            "solar_zero_corrected_hours": int((group["solar_mw"] == 0).sum()),
            "solar_wind_residual_gt_1mw": int((residual.loc[group.index].abs() > 1).sum()),
        }
        for measure in ["demand_mw", "wind_mw", "solar_mw", "total_gen_mw"]:
            row[f"{measure}_mean"] = float(group[measure].mean())
            row[f"{measure}_median"] = float(group[measure].median())
            row[f"{measure}_p01"] = float(group[measure].quantile(0.01))
            row[f"{measure}_p99"] = float(group[measure].quantile(0.99))
        era_rows.append(row)
    pd.DataFrame(era_rows).to_csv(OUT / "full_dataset_era_summary.csv", index=False)
    big_jumps.to_csv(OUT / "full_dataset_demand_jumps.csv", index=False)

    quality = {
        "source_path": str(DATA.relative_to(ROOT)),
        "rows": len(df),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "non_hourly_intervals": int((intervals != 3600).sum()),
        "expected_rows_from_range": int(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600 + 1
        ),
        "missing_by_column": {col: int(df[col].isna().sum()) for col in MEASURES},
        "nonpositive_by_column": {
            col: int((df[col] <= 0).sum()) for col in ["demand_mw", "wind_mw", "total_gen_mw"]
        },
        "negative_solar_raw_hours": int((df["solar_raw_mw"] < 0).sum()),
        "negative_solar_corrected_hours": int((df["solar_mw"] < 0).sum()),
        "all_zero_measure_hours": len(all_zero),
        "positive_solar_flatline_hours": int(solar_flatline.sum()),
        "incomplete_hours": int((df["rows_per_hour"] < df["expected_rows_per_hour"]).sum()),
        "overfull_hours": int((df["rows_per_hour"] > df["expected_rows_per_hour"]).sum()),
        "solar_wind_residual_gt_1mw": int((residual.abs() > 1).sum()),
        "solar_wind_max_abs_residual_mw": float(residual.abs().max()),
        "demand_jumps_over_40pct": len(big_jumps),
        "repeated_runs": {
            col: repeated_runs(df, col) for col in ["demand_mw", "wind_mw", "total_gen_mw"]
        },
    }
    with (OUT / "full_dataset_model_readiness.json").open("w") as handle:
        json.dump(quality, handle, indent=2)
    baseline = recheck_baselines()
    print(json.dumps(quality, indent=2))
    print("\nRenewable baseline recheck:\n", baseline.to_string(index=False))


if __name__ == "__main__":
    main()
