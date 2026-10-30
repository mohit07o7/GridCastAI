# January 10, 2025 demand spot-check reconciliation

Verified October 31, 2026. The spot-check is the **Desktop purged-split run**, while the audit workspace retains the earlier **unpurged-split run**. The source scripts differ only in the split slicing. Replaying both scripts in the same Python environment reproduces every one of their 4,986 saved test predictions within 0.00625 MW (CSV float formatting). Neither historical forecast was overwritten.

| Item | Workspace run | Desktop spot-check run |
|---|---:|---:|
| Target | 2025-01-10 15:00 | 2025-01-10 15:00 |
| Forecast origin | 2025-01-09 15:00 | 2025-01-09 15:00 |
| Actual demand (MW) | 207,588.72 | 207,588.72 |
| XGBoost (MW, full replay precision) | 202,060.5 | 201,367.78125 |
| Absolute error (MW) | 5,528.22 | 6,220.93875 |
| Prior-day baseline (MW) | 203,618.10 | 203,618.10 |
| Training rows | 23,268 | 23,244 |
| Validation rows | 4,986 | 4,962 |
| Test rows | 4,986 | 4,986 |
| Best iteration (zero-based) | 216 | 216 |
| Saved full-test MAE (MW) | 5,268.38 | 5,312.82 |
| Saved full-test MAPE (%) | 2.7056 | 2.7283 |

The exact forecast difference is 692.71875 MW. The displayed difference is 692.72 MW. This is a changed fitted model, not an arithmetic or target-alignment error. Both use 217 selected boosting rounds; early stopping selected the same iteration despite the changed training and validation samples.

## Timestamp and data lineage

The raw `Desktop/GridCastAI/data/raw/January 2024- June 2025.xlsx`, sheet `Report`, has 203,618.10 MW at the origin and 207,588.72 MW at the target. The script verifies demand, wind, and solar against both processed datasets at these two times. The modern workbook passes through `resample_to_hourly.py` without time aggregation, with solar clipped at zero. `prepare_demand_data.py` renames the timestamp, coerces numeric columns, sorts/deduplicates, and preserves demand. Both copies of `demand_model_data.csv` are byte-identical (SHA-256 in the JSON evidence).

The target is CSV line 1019 in both saved demand prediction files: origin January 9, target January 10. Selecting January 10 in **Forecast Origin** instead would select the forecast for January 11. All timestamps are naive source-clock timestamps; no timezone conversion is performed or inferred.

There are 33,432 input rows and 33,240 feature-complete labeled rows after 168 hours of warm-up and 24 hours of unavailable future labels. `Target = demand_mw.shift(-24)` associates the January 9 origin with January 10 demand.

## Exact feature vector

Lags are relative to the **origin**, not the target. Calendar fields also describe the origin (Thursday, day-of-week 3), not Friday (4).

| Feature | Value | Source timestamp or inclusive window |
|---|---:|---|
| demand_lag_1 | 203543.3 | Jan 9 14:00 |
| demand_lag_24 | 203214.2 | Jan 8 15:00 |
| demand_lag_48 | 205481.75 | Jan 7 15:00 |
| demand_lag_168 | 193562.8 | Jan 2 15:00 |
| demand_mean_24h | 207037.14083333334 | Jan 8 15:00–Jan 9 14:00 |
| demand_mean_168h | 184958.07428571428 | Jan 2 15:00–Jan 9 14:00 |
| wind_lag_1 | 3153.21 | Jan 9 14:00 |
| wind_lag_24 | 2499.62 | Jan 8 15:00 |
| solar_lag_1 | 49594.33 | Jan 9 14:00 |
| solar_lag_24 | 41850.32 | Jan 8 15:00 |
| hour | 15 | Jan 9 15:00 |
| day_of_week | 3 | Jan 9 15:00 |
| month | 1 | Jan 9 15:00 |

All dates above are in 2025. Rolling means use `shift(1).rolling(...)`, excluding the origin hour. The seasonal-naive forecast uses demand **at the origin**, not `demand_lag_24`: the latter is 48 hours before the target. The January frozen-demand episodes remain in the data and can affect rolling features, but they are identical inputs in both runs and do not explain this mismatch.

## Model configuration and boundary correction

Both scripts use `XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, early_stopping_rounds=50, eval_metric='mae', random_state=42, n_jobs=4)`. Both train only on training rows and use validation for early stopping; neither refits on training plus validation. No serialized demand model is present; reproduction requires training.

The workspace takes `[:train_end]`, `[train_end:val_end]`, and `[val_end:]`. The Desktop changes the first two to `[:train_end-24]` and `[train_end:val_end-24]`, preserving the test window.

- Workspace: latest training target **May 5 11:00**, after first validation origin **May 4 12:00**; latest validation target **November 29 05:00**, after first test origin **November 28 06:00** (all 2024).
- Desktop: latest training target **May 4 11:00**, before first validation origin **May 4 12:00**; latest validation target **November 28 05:00**, before first test origin **November 28 06:00**.

The Desktop purge therefore corrects label availability at the split boundaries. The original targets do not overlap each other across splits, but some labels were not yet observable at the next split's first forecast origin. Removing those rows changes tree fitting even with a fixed seed and identical hyperparameters. Use the purged run's 2.73% MAPE when describing this spot-check; the workspace README's 2.71% is explicitly a different historical run. This reconciliation does not resolve the separate source-quality limitations documented in the readiness audit.

## Reproduce safely

From any working directory, run:

```bash
/Users/mohityadav/Desktop/GridCastAI/.venv/bin/python \
  /Users/mohityadav/Documents/ChatGPT/GridCaseAi/src/investigations/reproduce_demand_spot_check.py \
  --target '2025-01-10 15:00:00' \
  --compare-root /Users/mohityadav/Desktop/GridCastAI \
  --raw-workbook '/Users/mohityadav/Desktop/GridCastAI/data/raw/January 2024- June 2025.xlsx' \
  --retrain \
  --output /Users/mohityadav/Documents/ChatGPT/GridCaseAi/reports/audits/demand_spot_check.json
```

Omit `--retrain` for a fast saved-row/data/feature trace. Omit `--compare-root` to inspect only the workspace. The script executes each trusted experiment's source only up to feature preparation or model fitting, never its forecast-writing section. It rejects missing/duplicate target rows, non-hour timestamps, timezone-aware requests, targets outside the test split, data alignment failures, and replay differences above 0.02 MW. The tolerance accommodates stored float formatting, not material forecast drift.

The JSON captures paths, input/script/forecast hashes, exact features and source times, split dates, raw values, parameters, library versions, and full-test replay agreement. Verified with Python 3.14.3, pandas 3.0.6, NumPy 2.5.3, XGBoost 3.4.1, and scikit-learn 1.9.1. Requirements currently specify minimum versions rather than a locked historical environment; future library changes may fail the replay check and should be investigated rather than silently accepted.
