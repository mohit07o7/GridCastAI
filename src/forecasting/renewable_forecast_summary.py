"""
renewable_forecast_summary.py
==============================
Produces a unified comparison table of all renewable forecasting results:
  - Seasonal-naive baselines (wind + solar)
  - XGBoost Wind v2
  - XGBoost Solar v2 (raw and clipped)
  - Error breakdown by time of day for solar

Reads from previously saved metrics and predictions CSVs.
Does not re-train any models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "reports" / "experiments"
OUTPUT_PATH = EXPERIMENTS / "renewable_forecast_comparison.csv"

NIGHT_HOURS = set(range(0, 6)) | set(range(21, 24))


def pct_improvement(baseline_mae, model_mae):
    return (baseline_mae - model_mae) / baseline_mae * 100


def main():
    print("=" * 72)
    print("GRIDCASTAI — RENEWABLE FORECAST COMPARISON SUMMARY")
    print("=" * 72)

    # ---- Wind -------------------------------------------------------
    wind_metrics = pd.read_csv(EXPERIMENTS / "wind_forecast_metrics.csv")
    wind_naive = wind_metrics[wind_metrics["Model"] == "Seasonal-naive"].iloc[0]
    wind_xgb = wind_metrics[wind_metrics["Model"].str.contains("XGBoost")].iloc[0]

    wind_pct = pct_improvement(wind_naive["MAE (MW)"], wind_xgb["MAE (MW)"])

    # ---- Solar (full set) -------------------------------------------
    solar_metrics = pd.read_csv(EXPERIMENTS / "solar_forecast_metrics.csv")
    solar_naive = solar_metrics[
        solar_metrics["Model"].str.contains("Seasonal-naive")
    ].iloc[0]
    solar_xgb_clipped = solar_metrics[
        solar_metrics["Model"].str.contains("clipped") &
        solar_metrics["Model"].str.contains("XGBoost")
    ].iloc[0]
    solar_pct = pct_improvement(
        solar_naive["MAE (MW)"], solar_xgb_clipped["MAE (MW)"]
    )

    # ---- Solar by time of day ---------------------------------------
    solar_preds = pd.read_csv(
        EXPERIMENTS / "solar_forecast_predictions.csv",
        parse_dates=["Target Timestamp"],
    )
    solar_preds["hour"] = solar_preds["Target Timestamp"].dt.hour
    solar_preds["is_night"] = solar_preds["hour"].isin(NIGHT_HOURS)

    def mae_group(group):
        return {
            "n": len(group),
            "actual_mean": group["Actual (MW)"].mean(),
            "naive_MAE": mean_absolute_error(
                group["Actual (MW)"],
                group["Seasonal-naive Prediction (MW)"],
            ),
            "xgb_clipped_MAE": mean_absolute_error(
                group["Actual (MW)"],
                group["XGBoost Clipped Prediction (MW)"],
            ),
        }

    day_grp = solar_preds[~solar_preds["is_night"]]
    night_grp = solar_preds[solar_preds["is_night"]]
    day_stats = mae_group(day_grp)
    night_stats = mae_group(night_grp)

    # ---- Print results ----------------------------------------------
    print()
    print("WIND (MW) — 24-hour ahead, Test set")
    print("-" * 50)
    print(f"  Seasonal-naive MAE :  {wind_naive['MAE (MW)']:>8,.1f} MW")
    print(f"  XGBoost v2 MAE     :  {wind_xgb['MAE (MW)']:>8,.1f} MW"
          f"  ({wind_pct:+.1f}% vs naive)")
    print(f"  Best iteration     :  {int(wind_xgb['Best Iteration'])}")
    print(f"  Test observations  :  {int(wind_xgb['Test Observations']):,}")

    print()
    print("SOLAR (MW corrected) — 24-hour ahead, Test set")
    print("-" * 50)
    print(f"  Seasonal-naive MAE (clipped) :  {solar_naive['MAE (MW)']:>8,.1f} MW")
    print(f"  XGBoost v2 MAE    (clipped)  :  "
          f"{solar_xgb_clipped['MAE (MW)']:>8,.1f} MW"
          f"  ({solar_pct:+.1f}% vs naive)")
    print()
    print("  By time of day:")
    print(f"    Daytime  (06-20):  Actual mean {day_stats['actual_mean']:>8,.0f} MW")
    print(f"                       Naive MAE   {day_stats['naive_MAE']:>8,.0f} MW")
    print(f"                       XGB MAE     {day_stats['xgb_clipped_MAE']:>8,.0f} MW")
    day_pct = pct_improvement(day_stats["naive_MAE"], day_stats["xgb_clipped_MAE"])
    print(f"                       vs naive    {day_pct:>+8.1f}%")
    print()
    print(f"    Night    (21-05):  Actual mean {night_stats['actual_mean']:>8,.0f} MW")
    print(f"                       Naive MAE   {night_stats['naive_MAE']:>8,.0f} MW")
    print(f"                       XGB MAE     {night_stats['xgb_clipped_MAE']:>8,.0f} MW")
    night_pct = pct_improvement(night_stats["naive_MAE"], night_stats["xgb_clipped_MAE"])
    print(f"                       vs naive    {night_pct:>+8.1f}%")

    print()
    print("DIAGNOSIS — Solar model underperformance")
    print("-" * 50)
    print("  The solar model's daytime MAE is significantly worse than the")
    print("  seasonal-naive baseline. Root cause: the test set (Apr–Jun 2025)")
    print("  has systematically higher peak irradiance than the training set")
    print("  (Jan 2024 – Jan 2025, dominated by winter months).")
    print()
    print("  Training mean solar:  ~13,937 MW  (Jan 2024 – Jan 2025)")
    print("  Test mean solar:      ~19,305 MW  (Apr 2025 – Jun 2025)")
    print()
    print("  The seasonal-naive baseline (same-hour yesterday) automatically")
    print("  tracks the current irradiance level, while XGBoost's lag-based")
    print("  features predict towards the lower training distribution.")
    print()
    print("  Solutions (in priority order):")
    print("  1. Extend data to >= 2 full years -> enables lag_8760 (same hour")
    print("     one year ago) which directly encodes year-on-year irradiance.")
    print("  2. Add NWP weather forecast features (GHI, cloud cover).")
    print("  3. Use a rolling origin validation strategy so that more recent")
    print("     months are always represented in training.")

    # ---- Save CSV ---------------------------------------------------
    rows = [
        {
            "Target": "Wind (MW)",
            "Model": "Seasonal-naive",
            "MAE (MW)": round(wind_naive["MAE (MW)"], 1),
            "RMSE (MW)": round(wind_naive["RMSE (MW)"], 1),
            "MAPE (%)": round(wind_naive["MAPE (%)"], 2),
            "vs Naive (%)": 0.0,
        },
        {
            "Target": "Wind (MW)",
            "Model": "XGBoost-Wind-v2",
            "MAE (MW)": round(wind_xgb["MAE (MW)"], 1),
            "RMSE (MW)": round(wind_xgb["RMSE (MW)"], 1),
            "MAPE (%)": round(wind_xgb["MAPE (%)"], 2),
            "vs Naive (%)": round(wind_pct, 2),
        },
        {
            "Target": "Solar (MW) corrected",
            "Model": "Seasonal-naive (clipped)",
            "MAE (MW)": round(solar_naive["MAE (MW)"], 1),
            "RMSE (MW)": round(solar_naive["RMSE (MW)"], 1),
            "MAPE (%)": round(solar_naive["MAPE (%)"], 2),
            "vs Naive (%)": 0.0,
        },
        {
            "Target": "Solar (MW) corrected",
            "Model": "XGBoost-Solar-v2 (clipped)",
            "MAE (MW)": round(solar_xgb_clipped["MAE (MW)"], 1),
            "RMSE (MW)": round(solar_xgb_clipped["RMSE (MW)"], 1),
            "MAPE (%)": round(solar_xgb_clipped["MAPE (%)"], 2),
            "vs Naive (%)": round(solar_pct, 2),
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Comparison table saved to: {OUTPUT_PATH}")
    print("  Raw data was not modified.")


if __name__ == "__main__":
    main()
