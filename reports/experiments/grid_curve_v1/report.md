# Verified 24-hour forecasting v1

Forecasting is complete. Battery implementation and verification are set aside at the user’s request. This report contains no battery result.

Results are retrospective: these historical periods have already informed development. The selected forecast and validation-selected baseline use identical trusted delivered hours within each comparison.

| Horizon | Paired hours | Net-demand forecast MAE | Baseline MAE | MAE reduction |
|---|---:|---:|---:|---:|
| 1 | 4924 | 2,546.04 MW | 5,392.01 MW | 52.78% |
| 6 | 4924 | 4,036.08 MW | 6,065.27 MW | 33.46% |
| 12 | 4924 | 4,538.76 MW | 6,063.15 MW | 25.14% |
| 24 | 4924 | 4,789.75 MW | 6,101.70 MW | 21.50% |

The delivered system lowers paired net-demand MAE at 24/24 horizons. This does not establish future performance or any battery benefit.
All 4,991 nominal issue times return a complete curve; 578 curves contain at least one labelled fallback.

## Monthly behavior

- Horizon 1: lower MAE in 8/8 months; worse months: none.
- Horizon 24: lower MAE in 8/8 months; worse months: none.

## How it works

- One issue returns 24 distinct hourly targets, origin + 1 through origin + 24. Source-clock timestamps remain timezone-naive, and the observation at the origin is assumed available.
- All 72 candidates were trained on training data only. Fixed chronological boundaries, a 24-hour purge, validation early stopping and validation-only method selection precede all test scoring. Validation selected XGBoost for every component/horizon; test scores did not change those choices.
- Lag/rolling features and target-calendar fields contain no future labels. Observed quality checks use evidence available by the origin; retrospective target exclusions attach only after generating predictions.
- Baselines are last observation, target-aligned previous day and previous week. If a selected method is unavailable, eligible baselines are explicit fallbacks; unsafe components remain null.
- The existing single-target command, older model bundles, source measurements and historical results remain unchanged. The new bundle stores native models, feature order/types, selected iterations, configuration, dates, row counts, versions and hashes.
- Selected-only and delivered cohorts are exported separately. MAE, RMSE, WAPE and signed error (forecast minus actual) are reported by component, horizon and month. A zero WAPE denominator is undefined.

## Reproduce

```bash
.venv/bin/python src/forecasting/predict_grid_curve.py --origin "2025-06-24 23:00:00" --history data/processed/full_hourly_stacked.csv
.venv/bin/python src/analysis/evaluate_grid_operations.py --forecast-only
.venv/bin/python src/investigations/verify_grid_curve.py
.venv/bin/python src/investigations/verify_grid_curve_evaluation.py
```

forecast_rows.csv.gz contains every forecast, method, quality status and retrospective exclusion. forecast_metrics.csv contains every paired metric. verification.json verifies native model reproduction; evaluation_verification.json independently verifies exported scores. No training occurs in prediction/evaluation/verification.
