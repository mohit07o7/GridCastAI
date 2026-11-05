# Verified 24-hour forecasts — battery work deferred

Run commands from `/Users/mohityadav/Desktop/GridCastAI`. These additions preserve the older command, which predicts one target exactly 24 hours ahead. The new command predicts **all 24 hours from one issue time**. Neither prediction command fetches live data.

Current delivery is forecasting only, following the user's scope update. Battery code and configuration are retained separately for later work; the full battery replay was stopped and its verification is deferred. No battery performance result is claimed.

## What was built, and why

1. **Direct forecasts:** 72 independently trained XGBoost candidates represent demand, wind and solar at horizons 1–24. Recursive predictions are not substituted for observations. Separate models let the relationship between history and the next hour differ from the relationship with tomorrow's same hour.
2. **Validation selection:** each candidate competes with last-observation, previous-day and previous-week forecasts on identical eligible validation rows. Baselines win ties. The selected method and baseline fallback ranking are saved, not reselected using test outcomes.
3. **Load-only prediction:** one cached `GridCurveForecaster` returns the complete curve and component-specific quality results. It imports no training entry point.
4. **Forecast evaluation:** score delivered forecasts and explicit fallbacks on paired trusted target rows, with selected-only results separately labelled. The forecast-only path disables both training and battery solving.

All historical results are retrospective. The historical test period has informed previous development; it is not a new untouched holdout.

## Commands

Train once into the new versioned bundle:

```bash
.venv/bin/python src/forecasting/train_grid_curve.py
```

The bundle already exists after this implementation. Training refuses to overwrite it. A deliberate retraining should use a new model version rather than replacing the verified bundle.

Load a curve ending beyond the available history:

```bash
.venv/bin/python src/forecasting/predict_grid_curve.py \
  --origin '2025-06-24 23:00:00' \
  --history data/processed/full_hourly_stacked.csv
```

This returns targets June 25 at 00:00 through 23:00. `--output /tmp/grid-curve.json` saves the exact JSON printed to stdout and refuses to overwrite an existing file. `--root` selects another project artifact root. The optional `--scenario` and `--soc-mwh` switches belong to the deferred battery work and are not needed for forecasting.

```bash
.venv/bin/python src/analysis/evaluate_grid_operations.py --forecast-only
.venv/bin/python src/investigations/verify_grid_curve.py
.venv/bin/python src/investigations/verify_grid_curve_evaluation.py
PYTHONPATH=src:tests .venv/bin/python -m unittest test_demand_model test_demand_pipeline test_grid_prediction test_grid_evaluation test_grid_curve
```

Forecast-only evaluation regenerates only its own versioned outputs; it disables XGBoost training entry points and the battery solver. Independent verification reconstructs forecast metrics from exported rows rather than trusting the scoring implementation.

## Forecast definitions and quality

Inputs accept `timestamp` or `Timestamp`, demand/wind/corrected solar in MW, and optional independently recorded `solar_wind_mw`. Timestamps must be unique hourly timezone-naive source-clock values. The origin observation is assumed available. Future observations are removed before measurement quality is inspected. Invalid timestamp syntax is an input-schema error.

For component value `y`, origin `t`, and horizon `h`:

- Label: `y(t+h)`, attached only in training or evaluation.
- Previous-day baseline: `y(t+h-24)`; previous-week: `y(t+h-168)`; last observation: `y(t)`.
- Inputs: lags 0, 1, 3, 6, 24, 48 and 168; 24/168-hour origin-including mean and sample standard deviation (`ddof=1`); target-aligned day/week observations; target hour, weekday and month sine/cosine, plus target year.
- Candidate: previous-day baseline plus a learned residual, floored at zero. A nonpositive demand model output is rejected in favor of an eligible baseline. Observations themselves are never clipped or repaired here.

Every model needs a complete 169-hour interval. The curve inference policy reuses observed simultaneous zeros, six-hour frozen runs, positive solar flatlines above 500 MW, demand changes above 40%, and optional renewable inconsistencies above 1 MW. It checks each model's own component; no cross-component lag features are introduced. A demand zero is unusable even when other components are nonzero.

A run cannot be recognized before its sixth available hourly reading. Once recognized, its earlier visible readings can also be rejected as baseline references. Batched historical replay applies the same detection time as a history prefix; eventual future run length does not change earlier decisions. Missing context is reported separately as an unavailable quality check.

For each component and horizon, use the selected method when eligible; otherwise try valid baselines in saved validation-score order. Before that validation information was available, use the fixed previous-day/last-observation/previous-week order, explicitly labelled as a fallback. Reject unsafe baseline references; do not assume a different historical timestamp is clean. A blocked component stays null.

Each point returns method, status, reasons, unavailable checks and model hash. `target_quality` remains `unknown`. `complete_curve` means all three components are delivered at every horizon. Exit codes are 0 for a usable/degraded curve, 2 when every target hour has at least one blocked component, and 1 for a request/schema error. An incomplete curve cannot produce a full battery plan.

## Training and split rules

The fixed raw-time boundaries are validation origin May 3, 2024 at 02:00 and test origin November 28, 2024 at 01:00. Before quality filtering, purge any training/validation origin whose maximum 24-hour target reaches the next split. Assert strict target-before-next-origin separation.

Fitting uses training rows only, seed 42, four threads, histogram trees, absolute-error loss, depth 4, learning rate 0.03, at most 1,000 trees and 50-round validation early stopping. Models are not refit on validation or test. Save and reuse the explicit selected iteration range.

Training target masks are component-specific retrospective masks. Demand removes invalid/nonpositive targets, all-zero records, April 4's known anomaly and frozen demand. Wind removes invalid targets, all-zero records, frozen wind and renewable inconsistencies. Solar removes invalid targets, all-zero records, positive flatlines and inconsistencies. Input eligibility uses the causal inference policy, separately from those label masks.

The new model bundle records native model hashes, feature names/types, configuration, selection scores, actual split ranges/counts, source and code hashes, package versions and full-precision test references. Code changes recorded by that bundle require a deliberately reviewed new version. A fresh-process verifier reproduces every saved candidate prediction within 0.000001 MW with fitting disabled.

## Deferred battery implementation

The following implementation was prepared before battery work was set aside. Its complete replay and independent physical/cost verification are not part of the current delivery. Do not interpret its code or examples as a validated battery result.

`configs/operations_v1.json` holds the explicit assumptions. All three national profiles use the same scale: training-period peak demand becomes 100 MW. Later demand can exceed 100 MW. The scale does not describe a real site's installed capacities.

Defaults are 80 MW import capacity; no export; 80 MWh battery; 20 MW charge/discharge power; charge and discharge efficiencies each sqrt(0.9); 8–72 MWh operating range; 40 MWh initial and terminal-reserve charge; and ₹250/MWh of total charging-plus-discharging throughput. These wear costs are illustrative, not a measured degradation model.

Synthetic electricity prices are ₹3,000/MWh at hours 00–05, ₹5,000 at 06–16 and 22–23, and ₹8,000 at 17–21, following the source clock. Every hourly MW reading is assumed constant for its simulated hour to obtain MWh. Source measurements do not independently establish interval energy or tariffs.

For each planned hour:

```text
grid + renewable used + discharge + unmet demand = demand + charge
next energy = current energy + eta_charge * charge - discharge / eta_discharge
```

Grid import, renewable use, battery energy and power are bounded. Binary direction prevents simultaneous charging/discharging. The optimizer uses SciPy MILP with a 10-second limit per pass and relative gap 1e-7; unsuccessful solves are unavailable plans, not silently accepted incumbents. Returned plans receive separate physical constraint checks (2e-5 MW/MWh numerical tolerance).

First minimize total unmet MWh. Then keep that optimum within numerical tolerance while minimizing purchases plus wear. Terminal energy must remain at least 50% of capacity. A plan with any material unmet demand is explicitly not fully served.

## Historical execution and comparisons

Forecast all nominal origins before attaching future target actuals. Current data supplies 4,991 full-24-hour origins, November 28, 2024 at 01:00 through June 23, 2025 at 23:00. Physical execution scores their first hours, through June 24 at 00:00; it does not pretend to observe future hours beyond the dataset.

Compare no battery, baseline-driven battery planning and selected-forecast battery planning. Both battery controllers start with the same charge but subsequently keep independent trajectories. At execution, reduce charging that supply cannot support and reduce discharge that would create surplus; never increase the planned discharge using hindsight. Use actual renewables first, import up to the limit, curtail remaining surplus and record remaining shortage. The reconciled charge feeds the next decision.

Apply the documented combined retrospective target policy only for evaluation: invalid/negative actuals, nonpositive demand, all-zero records, April 4 anomaly, frozen demand/wind, positive solar flatlines and recorded renewable inconsistencies. Plans are produced before checking the current target's flag. An untrusted actual breaks the physical episode, because a trustworthy energy transition cannot be reconstructed. The next trusted episode restarts at 50% charge for all battery controllers. This explicit retrospective reset convention is not an online state estimator.

Forecast or solver failures idle the battery and remain in the scored trusted-hour cohort. Report coverage and service as well as cost; supplying less energy is not automatically a useful saving. Compare grid limits 70/80/100 MW and energy capacities 40/80/160 MWh separately. Capacity changes preserve SOC fractions and leave power at 20 MW.

There is no real tariff, network model, thermal dispatch, forecast-error scenario optimizer, or deployable battery controller in this version. Aggregate reported renewable generation is a simulated availability signal rather than independently measured unconstrained renewable potential. Those additions need separate data and evaluation.

## Results and reusable interfaces

Read `reports/experiments/grid_curve_v1/report.md` for the verified forecast conclusion. Compressed forecast rows preserve every timestamp, component method, quality status and exclusion. The metrics cover each horizon and month. Battery results remain deferred.

With `src` on the Python path:

```python
from common.curve_model import GridCurveForecaster

service = GridCurveForecaster()  # reuse; no fit
curve = service.forecast(history, origin)
```

The batch `forecast_rows` path is for efficient causal historical replay. It accepts a normalized `CurveHistory` and observed origins, returns no future labels, and is checked against prefix-based prediction.

## Inspect quality examples

`reports/experiments/grid_operations_v1/examples.json` records a normal end-of-history curve, a labelled historical fallback at January 9, 2025 at 15:00, and an explicitly synthetic blocked-wind example. The latter sets wind negative in a temporary 169-hour fixture, removes the optional combined-renewable channel, and checks that demand and solar continue while the full battery plan is unavailable. No source measurements are edited. The independent command verifier recreates these cases with training disabled.
