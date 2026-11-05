# GridCastAI local dashboard and demonstration

The dashboard displays the verified direct forecasts for horizons 1–24, historical accuracy, origin-time input quality, retrospective exclusions, and renewable contribution. **Optimization is Deferred.** No battery schedule, dispatch or savings claim is generated.

Everything runs locally in `/Users/mohityadav/Desktop/GridCastAI`. The application does not train, replay experiments or install packages on startup. The original model artifacts, training code, measurements and historical reports are preserved.

## Install once

Use the existing tested Python 3.14.3 project environment:

```bash
cd /Users/mohityadav/Desktop/GridCastAI
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python -m pip check
```

The separate dashboard dependency file inherits the original numerical pins unchanged. Its reviewed additions are pinned exactly, including Streamlit 1.66.1 and Plotly 7.1.0. This is a selected dependency list, not an unrelated environment freeze. Dependency installation needs network access; the installed dashboard uses local data and models.

## Start the dashboard

```bash
./scripts/start_dashboard.sh
```

Open **http://127.0.0.1:8501**. The launcher resolves the project root even when called from another directory. It uses the project's `.venv`, performs artifact preflight and binds only to the local interface. Training and battery solver entry points are disabled in the server process.

For a preflight-only check or another local port:

```bash
./scripts/start_dashboard.sh --check
./scripts/start_dashboard.sh --port 8502
```

Stop the foreground server with Ctrl+C. An occupied port should be resolved by stopping the existing dashboard process or selecting another port. Model outages leave the historical views accessible when their own verified files are available. Missing dependencies produce the installation command; the launcher never installs them automatically.

## Five-minute walkthrough

1. **24-hour forecast:** the initial origin is June 24, 2025 at 23:00. Its target times are June 25 at 00:00–23:00. Review aligned demand, wind and solar, renewable generation, signed net demand, and renewable share. Actual readings beyond the data end are unavailable.
2. **Historical fallback:** click the sidebar preset. January 9 at 15:00 produces a complete curve with labelled baseline fallbacks. Optional actual overlays are attached after prediction. They do not become features or change `target_quality: unknown`.
3. **Synthetic blocked wind:** the example copies a 169-hour source prefix in memory, removes the optional combined-renewable channel and replaces wind with invalid negative values. Wind is blocked while demand and solar continue. Wind, combined renewable and net-demand gaps remain missing; the source file is untouched.
4. **Historical accuracy:** default selection is net demand, horizon 24, delivered cohort and all months. Change component, horizon, cohort or month. Every metric comes from the matching saved paired row. Monthly bars use each month's own identical model/baseline mask. A changed filter does not issue a new forecast.
5. **Quality and exclusions:** observed input problems for the displayed forecast are separate from the retrospective target audit. The audit follows the accuracy view's horizon/month/cohort filters. Exclusion reasons can overlap; use the explicit excluded-record count rather than adding reason counts. Download all filtered exclusions.
6. **Optimization status:** Deferred. Signed net demand is a residual requirement calculation, not a dispatch solution.

Date/hour edits retain the displayed forecast until **Generate forecast** is pressed. Preset buttons both select and generate their example. The displayed heading always identifies the result's actual origin. If artifacts change after generation, a warning requests a fresh result.

CSV download contains exactly the 24 aligned forecast rows, method/status/reason fields and derived quantities shown by the service. JSON is the original forecasting-service result. Historical overlays remain separate and do not modify that JSON. Renewable share is undefined for nonpositive demand, is not capped at 100%, and signed net demand is not clipped at zero. Renewable-share differences against actuals use percentage points (pp).

## Reproduce a complete sample run

```bash
.venv/bin/python scripts/run_dashboard_demo.py
```

Each invocation creates a new `reports/demos/dashboard_v1/run-*` folder. It contains normal/fallback/blocked JSON and CSV files, the selected historical accuracy row, an April exclusion audit, a walkthrough and a hashed `run.json` receipt. The same service layer serves both this command and the dashboard. Training and battery entry points are disabled. It never writes source data or overwrites another run.

An explicit output folder can be supplied with `--output`; it must not already exist. The blocked case is synthetic and labelled as such.

## Tests and real-browser checks

Run all forecasting and dashboard regression checks, without deferred battery tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests .venv/bin/python -m unittest \
  test_demand_model test_demand_pipeline test_grid_prediction \
  test_grid_evaluation test_grid_curve test_dashboard_service test_dashboard_app
.venv/bin/python src/investigations/verify_grid_curve.py
.venv/bin/python src/investigations/verify_demand_model.py
.venv/bin/python src/investigations/verify_grid_prediction.py
```

Service tests check CLI parity within 0.000001 MW, future invariance, arithmetic/missing values, source/metric integrity and downloads. Streamlit AppTest exercises filters, retained forecasts, presets, overlays and historical access during a model outage. Browser tests check actual rendered charts and downloaded file bytes.

To reproduce Chrome screenshots and browser checks, install the optional test driver separately and start the dashboard first:

```bash
npm install --prefix /tmp/gridcast-dashboard-browser playwright@1.64.1
.venv/bin/python scripts/run_dashboard_demo.py
node scripts/check_dashboard_browser.cjs
```

The browser check uses the latest successful sample run and saves a new `browser-*` directory with screenshots and a verification receipt. It expects the standard macOS Google Chrome executable. Optional environment variables are `GRIDCAST_CHROME` (browser executable), `GRIDCAST_PLAYWRIGHT` (driver module path), and `GRIDCAST_DEMO_RUN` (specific sample folder). The browser check neither launches an externally hosted service nor installs a browser automatically.

## Architecture and performance

`src/dashboard/service.py` owns artifact validation, table construction and historical selection. It never imports training entry points. The interface calls the existing `GridCurveForecaster`; it does not reproduce feature definitions or change saved selections.

Streamlit caches the forecaster as a shared read-only resource and source/report tables as data. Cache keys include file path, size and nanosecond modification time; hashes are checked again on cache misses. The model bundle and historical reports validate separately, so a model failure does not hide valid saved history. Forecast generation occurs at initial load or explicit requests, while filters and downloads reuse the displayed result.

Historical accuracy is a retrospective development evaluation, not a new untouched holdout or a guarantee of future performance. Counts identify origin–horizon forecast records separately from unique target hours, especially in the all-horizon provenance totals.

See `reports/demos/dashboard_v1/verification.md` for the completed run, screenshots, dependency review, regression results and protected-file checks.
