# Renewable Forecasting v3: Four-Year Experiment

**Run date:** 21 October 2026  
**Input:** `data/processed/full_hourly_stacked.csv` (33,432 hourly records)  
**Code:** `src/forecasting/renewable_v3_full_history.py`  
**Status:** Wind residual model improves on persistence; solar persistence remains the recommended forecast for the measured target.

## Forecast definition and split

At forecast origin hour `t`, the observation for `t` is assumed available. The model predicts generation at `t+24`. The true same-hour-yesterday baseline is therefore the value at `t`, and residual XGBoost learns `generation(t+24) - generation(t)`. The saved older renewable scripts used `t-24` as their baseline, which is two days before the target.

The four-year stack is split chronologically by origin position: 70% for initial training, 15% for validation, and 15% for final test. A 24-hour origin gap precedes each next split. Internal assertions verify that the latest target used in training precedes the first validation origin, and the latest validation target precedes the first test origin. Model iteration is selected on validation; the final XGBoost candidate is refit using eligible training plus validation rows before the test prediction. Three earlier expanding-origin folds provide a stability check.

The test target window is **2024-11-29 01:00 to 2025-06-24 23:00**. Target and feature hours are removed if they encounter one of 12 all-zero source hours or the two October 2022 component dropouts. Solar additionally excludes the source-confirmed **48-hour 64,300.61 MW flatline** and any feature window made incomplete by it. This leaves **4,991 wind** and **4,751 solar** test predictions. Wind frozen target periods are excluded from fitting, while results are reported both with and without them.

## Candidate results on the eligible test rows

| Target | Candidate | Test observations | MAE | RMSE | WAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Wind | Prior-day persistence | 4,991 | 1,956.3 MW | 2,555.2 MW | 21.69% |
| Wind | Residual XGBoost | 4,991 | **1,894.9 MW** | **2,480.8 MW** | **21.01%** |
| Solar | Prior-day persistence | 4,751 | **1,154.6 MW** | 2,408.8 MW | **6.48%** |
| Solar | Daytime residual XGBoost, prior-day night | 4,751 | 1,160.7 MW | **2,394.1 MW** | 6.51% |
| Solar | Daytime residual XGBoost, hard-zero night | 4,751 | 1,194.6 MW | 2,412.0 MW | 6.70% |

Wind XGBoost reduces MAE by **61.3 MW (3.1%)** against the correctly aligned baseline. Its three earlier rolling folds each improve MAE by 90.6–156.7 MW. The validation-period paired day-block bootstrap also favors XGBoost: mean MAE improvement 109.7 MW, approximate 95% interval 56.2–168.1 MW. The saved wind model is the selected v3 candidate.

The wind gain varies by season. On the April–June 2025 overlap with the older renewable test, v3 wind scores **2,319.1 MW MAE** versus **2,322.6 MW** for persistence: only a **3.5 MW** gain. Monthly diagnostics show wind XGBoost worse in May and June 2025. The overall gain therefore does not establish a reliable summer/monsoon improvement; `renewable_v3_monthly_metrics.csv` records the month-by-month result.

Solar is mixed. The daytime residual model improves MAE by 28.4 MW on the final validation window but only one of three earlier rolling folds improves. On the final test it is **6.2 MW worse overall** than prior-day persistence, although its RMSE is slightly lower. The validation rule requires improvement in at least two of three earlier folds and a positive lower bootstrap bound; it therefore selects **prior-day persistence for solar**. This selection rule and the flatline mask were finalized after inspecting an initial test run, so the final solar test is **exploratory**, not a fresh untouched holdout. Confirmation needs new future data or a newly sequestered period.

## Solar-specific decisions

The daylight residual model is trained only on **06:00–20:00** target hours. It receives target-hour solar elevation and a nonnegative solar-shape proxy calculated at latitude 23°N, longitude 80°E in IST. These are approximate geometry features for a national aggregate; without plant locations, capacity, and cloud cover, they cannot define a defensible MW upper bound.

Hard-zero output for 21:00–05:00 was evaluated but not selected. The measured corrected solar series has positive values in many of those hours, so zeroing predictions raises error for this **recorded telemetry target**. A physically defined PV-only target would require a separate source definition and data audit before hard-zero output is appropriate.

Weather forecasts were not integrated. Historical reanalysis alone would leak future conditions if used as though available at issue time. A later experiment should join forecast weather by **issue timestamp** and location, then test whether it improves solar and wind predictions on a fresh period.

## Artifacts and reproducibility

- `reports/experiments/renewable_v3_metrics.csv`: validation and test metrics by target and segment.
- `reports/experiments/renewable_v3_rolling_folds.csv`: three earlier fold comparisons.
- `reports/experiments/renewable_v3_monthly_metrics.csv`: test-period monthly baseline and selected forecast scores.
- `reports/experiments/renewable_v3_wind_predictions.csv` and `renewable_v3_solar_predictions.csv`: row-level actuals, baselines, candidates, and selected forecasts.
- `reports/experiments/renewable_v3_metadata.json`: features, split dates, eligibility counts, model selection, and anomaly counts.
- `models/renewable_v3_wind.json` and `renewable_v3_solar.json`: saved XGBoost candidates; the solar candidate is experimental, while selected solar output is persistence.

Run `python src/forecasting/renewable_v3_full_history.py` from the project root with the dependencies in `requirements.txt`. The script writes new v3 artifacts and does not overwrite the earlier v2 experiment files or source measurements. The original v2 and v3 test periods differ, so their headline MAEs are not directly comparable.
