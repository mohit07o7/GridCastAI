# One command for aligned demand, wind and solar predictions

This command uses saved artifacts and observed history to predict **one target hour, exactly 24 hours after the origin**. It does not train, fetch live observations, produce a 24-hour curve, or optimize dispatch.

## Run it

From `/Users/mohityadav/Desktop/GridCastAI`:

```bash
.venv/bin/python src/forecasting/predict_grid.py \
  --origin '2025-06-24 23:00:00' \
  --history data/processed/full_hourly_stacked.csv
```

Add `--output /tmp/gridcast-forecast.json` to save exactly the JSON printed to stdout. Paths supplied on the command line are relative to your shell directory; saved model paths default to this project's location regardless of shell directory. `--root` selects another project artifact root. Output may not overwrite the history input or existing model artifacts. Output directories must already exist.

The expected normal target is `2025-06-25 23:00:00`, beyond the end of the example history. No future actual demand, wind or solar value is needed.

## What happens, and why

1. **Establish the origin.** Timestamp values must be valid, exact hourly, timezone-naive source-clock values. The origin reading is assumed available. No UTC or IST conversion is applied; the original source convention is not independently established by this command.
2. **Remove future observations.** Future measurement values, future duplicates and future anomaly episodes are never inspected. Timestamp syntax and agreement between two timestamp columns are schema checks on the input file; invalid timestamps are rejected because they cannot be assigned safely to either side of the cutoff.
3. **Build only past-derived features.** Demand uses the saved origin-relative lags, origin calendar and shifted means. Wind uses origin-including means/std, target calendar, and a point 8,736 hours before origin. No labels, target quality flags, or training-frame builders are used by production inference.
4. **Assess observed input quality.** The separately versioned `inference_quality_v1` policy reports suspicious inputs using only the origin-truncated history. This is not the historical scoring policy and does not determine whether an unknown future actual is good.
5. **Load the selected models.** Demand uses the saved `demand_v1` bundle and its selected 185 boosting rounds. Wind uses the saved 82-round v3 residual model, then adds the origin wind reading. Solar's selected method is origin-value persistence, including at night; the experimental solar model is not loaded.
6. **Fall back explicitly.** If history, quality or artifact checks prevent a selected model forecast, use origin-value persistence only if that origin reading passes available checks. Otherwise return a null forecast for that component. Other components continue.

## History and quality rules

Required columns are `Timestamp` or `timestamp`, `demand_mw`, `wind_mw`, and corrected nonnegative `solar_mw`. Both timestamp columns are allowed only if they agree. Input rows may be shuffled. Numeric strings can be parsed; invalid numbers become unavailable measurements for the affected dependencies. Negative, infinite, missing, boolean or complex measurements are invalid; values are never silently clipped or filled.

Demand needs the inclusive 169-hour interval [origin−168, origin], with valid demand values and valid wind/solar values at origin−1 and origin−24. Its own quality checks cover the interval; renewable flags are checked at the exogenous feature timestamps. A wind issue at the origin alone does not block demand, whose wind inputs precede that hour.

Wind needs its 169-hour recent interval plus the annual reference point. A fully continuous year between that point and the recent window is not required: those intervening hours are not model inputs. Missing required recent hours, a missing annual point, or flagged wind dependencies trigger persistence if the origin is usable.

Observed quality flags are:

- Simultaneous zero demand/wind/solar records, where all three measurements are available.
- Maximal constant-value runs with at least six consecutive hourly readings visible by the origin. Gaps break runs. Solar runs are flagged only above 500 MW, so normal zero solar at night is allowed. Once six readings have arrived, earlier readings in that observed run can also be recognised as suspect; future observations are never used.
- Demand changes strictly greater than 40% from a positive finite previous-hour value. There is no retrospective hard-coded April 4 date rule.
- Wind/solar inconsistency strictly above 1 MW when optional `solar_wind_mw` is available. No combined channel is reconstructed and treated as an independent measurement.

Quality checks with insufficient context are listed in `unavailable_quality_checks`; absence of a flag is not proof of perfect data. In particular, short inputs may permit persistence while lacking enough observations for a six-hour run check. Failed origin checks prohibit persistence. Component-specific outages and anomalies cannot all be identified without more source signals.

## Read the output

Each component returns the common target, forecast MW or null, actual method, status, reasons, unavailable checks, selected model ID/hash and artifact-validation status. `selected_model_id` identifies the configured model; when `method` is `prior_day_persistence` and `status` is `fallback`, that selected XGBoost model was not used.

- `ok`: selected method succeeded. For solar, persistence is the selected method.
- `fallback`: demand or wind used labelled origin-value persistence.
- `blocked`: no acceptable forecast is available for that component.

Overall `ok` requires all components to be `ok`. Overall `blocked` means all three are blocked. Any other combination is `degraded`. The CLI exits 0 for `ok` or `degraded`, 2 when all components are blocked, and 1 for invalid requests or file/schema errors. Error JSON is written to stderr.

`target_quality` is always `unknown`. The simulation label prevents historical outputs from being presented as forecasts actually issued on their historical dates. Demand/wind models are unavailable at origins earlier than their latest training/tuning target; persistence can still be used because it has no fitted parameters. The fixed solar baseline likewise does not require future learned parameters.

## Examples to check

Normal selected forecasts:

```bash
.venv/bin/python src/forecasting/predict_grid.py --origin '2025-06-24 23:00:00' --history data/processed/full_hourly_stacked.csv
```

Labelled demand fallback due to frozen readings in the recent history:

```bash
.venv/bin/python src/forecasting/predict_grid.py --origin '2025-01-09 15:00:00' --history data/processed/full_hourly_stacked.csv
```

The target is January 10 at 15:00. Demand persistence is 203,618.10 MW. This is not the historical XGBoost diagnostic of 202,477.65625 MW. The target remains excluded from the existing historical demand scoring policy.

Blocked demand component because the origin itself is in a visible frozen run:

```bash
.venv/bin/python src/forecasting/predict_grid.py --origin '2025-01-09 02:00:00' --history data/processed/full_hourly_stacked.csv
```

## Reuse in Python

With `src` on the import path:

```python
import pandas as pd
from common.grid_prediction import GridForecaster, predict_grid
history = pd.read_csv('data/processed/full_hourly_stacked.csv')
result = predict_grid(history, '2025-06-24 23:00:00')
# Reuse an instance for many origins to avoid reloading model artifacts.
forecaster = GridForecaster()
result = forecaster.forecast(history, '2025-06-24 23:00:00')
```

Features are independently implemented and checked against the original training definitions. Existing training code, demand bundle hashes, renewable models, experiment CSVs and unrelated files are preserved. The forecast manifest pins artifact hashes and records selected methods, feature definitions, and training/tuning cutoffs; regenerate and review it deliberately when replacing models. Integrity checks detect accidental mismatches, not malicious file replacement. No automatic retraining or silent alternate model selection occurs.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python src/investigations/verify_grid_prediction.py --output reports/audits/grid_prediction_verification.json
.venv/bin/python src/investigations/verify_demand_model.py
```

The integration verifier disables XGBoost fit/train/update entry points. Production inference never imports historical training builders; the verifier imports them solely as independent feature references. It compares features and numerical predictions across all 4,603 saved demand rows and 4,991 saved wind rows, and exercises the quality-aware service at 32 seasonal origins. It also verifies future-data invariance and CLI output/error behavior. New inference quality rules can cause fallbacks on historically scored rows; these checks do not create a new accuracy benchmark.
