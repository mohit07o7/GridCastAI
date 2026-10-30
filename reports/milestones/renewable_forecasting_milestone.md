# Milestone 2: Wind and Solar Forecasting Review

**Project:** GridCastAI  
**Review date:** 21 October 2026  
**Status:** Existing experiments reviewed; benchmark correction required before claiming model improvement.

## Scope and source

This report reviews the saved 24-hour-ahead wind and solar experiments trained on the **2024-01-01 to 2025-06-24** hourly dataset. It does not report a new training run on the four-year stack. The original metrics and predictions are in `reports/experiments/wind_forecast_metrics.csv`, `solar_forecast_metrics.csv`, `wind_forecast_predictions.csv`, and `solar_forecast_predictions.csv`. The comparisons below were recalculated by `src/auditing/audit_full_model_readiness.py` and saved to `reports/audits/renewable_baseline_recheck.csv`.

Each model used a chronological 70%/15%/15% origin-row split, with lag and rolling features. After feature and horizon trimming, the wind experiment had 8,920 train, 1,948 validation, and 1,924 test predictions; solar had 8,752 train, 1,948 validation, and 1,924 test predictions. Both saved test series cover target timestamps **2025-04-05 20:00 through 2025-06-24 23:00**.

## What the saved experiment reported

| Target | Saved “seasonal naive” MAE | XGBoost MAE | Saved comparison |
| --- | ---: | ---: | ---: |
| Wind | 3,255.8 MW | 2,650.0 MW | XGBoost 18.61% lower MAE |
| Solar, nonnegative output | 1,391.4 MW | 2,731.6 MW | XGBoost 96.32% higher MAE |

The solar model clipped negative predictions to zero. It still had **4,258.2 MW MAE during 06:00–20:00** on the full test set and a **3,444.7 MW average shortfall** during those hours. The test observations around noon are higher than the training observations in `reports/experiments/solar_distribution_shift.csv` (mean at hour 12: 56,789 MW test versus 44,867 MW train). A distribution shift is plausible, but the saved outputs alone do not establish its cause.

The original solar MAPE values are unsuitable as headline scores: actual solar output is zero or close to zero for some hours, so percentage errors become unbounded or extremely large. MAE, RMSE, and WAPE are reported instead. Even the 21:00–05:00 segment contains positive recorded solar values, so an hour-based night rule cannot be treated as a verified zero-output label.

## Baseline correction and current conclusion

The forecasting scripts define the target as the value **24 hours after** origin time `t`, but their saved baseline uses `lag_24` at `t`. That value is **48 hours before the target**, although it is labeled “same-hour yesterday.” A true prior-day baseline for target time `τ` uses the observed value at `τ - 24 hours`, which is the value at forecast origin `t`. The audit rejoined the saved target timestamps to `data/processed/hourly_grid_data_validated.csv` and verified that the saved actuals match the source before computing this correction.

| Target and test segment | True prior-day MAE | XGBoost MAE | XGBoost versus true prior day |
| --- | ---: | ---: | ---: |
| Wind, all 1,924 hours | **2,322.6 MW** | 2,650.0 MW | **14.1% higher MAE** |
| Solar, all 1,924 hours | **1,131.4 MW** | 2,731.6 MW | **141.4% higher MAE** |
| Solar, 06:00–20:00 (1,201 hours) | **1,728.1 MW** | 4,258.2 MW | **146.4% higher MAE** |

Thus the saved XGBoost models do **not** beat a correctly aligned prior-day baseline on these test observations. The wind model's 18.61% improvement claim only holds against the older, 48-hour baseline. Wind XGBoost's mean error (actual minus prediction) is **+1,648.2 MW**, and solar XGBoost's is **+2,203.6 MW**, indicating underprediction on average. These findings should replace the earlier renewable performance claim wherever it is presented.

## Split and data limitations

The scripts split by **forecast origin**, then shift targets forward 24 hours. The final 24 training origins therefore use outcomes from the next origin window, and the final 24 validation origins use outcomes from the test-origin window. For a strict walk-forward experiment, split eligibility must be defined by when each target becomes observable, or the earlier split must end at least 24 hours before the next split's first origin. The saved test metrics remain a useful historical diagnostic, but the experiment should be rerun with that boundary rule.

Wind and total generation contain documented frozen telemetry episodes; the expanded four-year audit found **11 runs of at least six identical hours** in each series, including an 87-hour run starting 2024-06-06 18:00. Anomaly sensitivity should be reported separately from the primary unaltered test. The full-dataset readiness report details additional legacy-source quality flags.

## Next experiment

1. Define forecast origin as the latest observation available when the prediction is made; document whether the current origin-hour reading is available. Use the same target timestamps and eligibility for every model and baseline.
2. Rebuild features on the four-year stack with source-quality flags. Never use post-origin readings or model labels as input features. Keep a fixed final chronological holdout and use rolling-origin validation within the earlier period.
3. Compare true prior-day persistence, weekly persistence, a simple seasonal baseline, and XGBoost on identical rows. Report MAE and RMSE for wind, and MAE/RMSE/WAPE plus a daylight segment for solar. Report mean error and error by month/hour to expose bias.
4. Treat weather forecasts and capacity changes as candidate features only after their issue times and provenance are documented. A year-ago lag is now feasible, but must prove useful against recent persistence rather than being assumed to improve accuracy.

The next milestone is complete only when the corrected baseline, clean split boundaries, anomaly treatment, and full-period source audit are reproducible from saved code and outputs.
