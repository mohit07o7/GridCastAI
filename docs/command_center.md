# GridCastAI cinematic command center

The new React/TypeScript application runs independently from the preserved Streamlit dashboard. It uses the existing verified Python prediction service and saved evaluation results through a thin local API. No models, feature definitions, training algorithms or source observations are rewritten.

## One-time setup

From `/Users/mohityadav/Desktop/GridCastAI`:

```bash
.venv/bin/python -m pip install -r requirements-command-center.txt
.venv/bin/python -m pip check
cd frontend
npm ci
npm run build
cd ..
```

Tested environment: CPython **3.14.3**, Node **25.1.0**, npm **11.6.2**. Existing numerical-package pins are inherited unchanged. The API uses Starlette **1.10.0** and Uvicorn **0.54.0**; it does not depend on Streamlit. Frontend direct versions and all transitive dependencies are recorded in `frontend/package.json` and `frontend/package-lock.json`. Fonts, India geometry, chart assets and map worker are bundled locally.

## Start

```bash
./scripts/start_command_center.sh
```

Open **http://127.0.0.1:8502**. This starts one Python process serving both the production React build and `/api/v1/*`. It performs no installation, training, experiment replay or battery solve. Existing Streamlit files and its separate launcher are unchanged.

```bash
./scripts/start_command_center.sh --check
./scripts/start_command_center.sh --port 8503
```

The script resolves its project root when called from another directory. Stop the foreground server with Ctrl+C. An existing process on the selected port must be stopped or a different port selected. Only loopback is supported; this is not a hosted deployment.

For frontend development, run the API on its default port and then `npm run dev` from `frontend`; Vite proxies `/api` to the local backend. Rebuild before using the production launcher. No wide-open cross-origin policy is needed.

## Application walkthrough

The default origin is **June 24, 2025 at 23:00**, returning June 25's 24 direct hourly targets. Horizon 24 is initially highlighted. All times describing observations and forecasts use the unchanged timezone-naive source clock.

1. **Command Center:** inspect national demand, wind, solar and signed net requirement for the highlighted horizon. Pan/zoom/reset the India reference map. The slider changes the highlighted target, not the forecast issue. Companion charts synchronize their target-time tooltips.
2. **Grid Demand Intelligence:** review the preceding reported history, the direct forecast curve, peak/minimum/change indicators and historical evaluation. Actual overlays are attached only after prediction.
3. **Renewable Energy Monitoring:** review separate wind/solar forecast, reported history and paired performance. Solar is not forced to zero by a decorative nighttime rule.
4. **AI Forecasting Lab:** set the origin in the shared issue bar and press Run forecast. Coverage is national only; the available bundle is `grid_curve_v1`. Normal, historical fallback and synthetic blocked-wind examples are reproducible. JSON is the original prediction result; CSV preserves all 24 flattened rows and their quality fields.
5. **Grid Requirement Analysis:** the arithmetic breakdown is demand minus wind minus solar. Signed surplus and renewable shares above 100% remain visible. This is a residual-requirement calculation, not physical dispatch.
6. **Model Intelligence:** choose quantity, horizon, cohort and month. MAE/RMSE/WAPE/signed error come from saved paired metrics. MAPE is calculated separately from the same scored rows after excluding zero actuals for both methods; the denominator count is displayed. Feature-gain importance is read from the native booster's selected prediction trees, not unused early-stopping trees. It is not a causal explanation.
7. **Historical Data Explorer:** apply a date range, inspect reported demand/renewables, retrospective flags and signed net demand. The API pages at most 5,000 rows per response; the interface retrieves the full selected range. Chart sampling is display-only. CSV downloads retain all selected source rows.
8. **System Operations:** inspect actual connectivity, artifact availability, source range, completed requests and measured durations. Logs contain only real events from the current API session and reset on restart. The target-exclusion audit is separate from issue-time input quality; reason counts can overlap.

Changing draft controls does not replace the displayed result until Run forecast succeeds. A failed request retains the previous result with an explicit message. Models and reports validate independently, so a model outage does not hide valid historical evidence. Synthetic input labels remain visible across all routes.

## Interpretation boundaries

- All forecasts and metrics are **retrospective research demonstrations**, not national live telemetry or forecasts actually issued on the historical dates.
- The map is a public-domain **Natural Earth geographic reference**. National aggregate coverage does not establish state/city measurements, substations, transmission routes or power flows. No such readings are fabricated. The map becomes a static outline if interactive rendering is unavailable.
- `target_quality` stays unknown during forecasting. Retrospective target flags attach only to historical exploration/evaluation.
- Blocked components are JSON nulls and chart gaps, never silent zeros. Nonpositive demand makes renewable share undefined. Undefined ratios remain missing.
- Model and baseline scores share identical eligibility masks. Counts distinguish forecast records from unique target hours. Current metric filters operate on one horizon at a time.
- Battery optimization remains **Deferred**. No battery actions, dispatch schedules or savings estimates are generated.

## Architecture and API

`src/command_api/app.py` adapts the existing read-only `dashboard.service` layer, which has no Streamlit imports. Resource caches use file signatures and validate hashes on reload. Historical quality annotation is cached separately. Inference is serialized in a worker thread; read/status endpoints remain available. Startup disables XGBoost fitting/native update and battery solver entry points.

The frontend uses React, strict TypeScript, Vite, Tailwind CSS, Framer Motion, Apache ECharts, MapLibre GL JS, Lucide, React Router, Zustand, TanStack Query and Axios. Zustand holds shared issue/horizon/filter state; TanStack Query caches typed HTTP responses. Map and chart code are lazy-loaded. Chart instances resize with their containers and dispose on unmount. Reduced motion and hidden-page behavior limit nonessential animation.

Endpoints:

| Method and path | Behavior |
|---|---|
| `GET /api/v1/status` | Actual connection/resource state, source range, evaluation months, uptime and last successful forecast |
| `POST /api/v1/forecast` | Body `{ "origin": "2025-06-24 23:00:00", "coverage": "IN-national" }`; returns original forecast, a 24-row presentation table and measured inference duration |
| `GET /api/v1/history` | `start`, `end`, `offset`, `limit` (1–5000); `format=csv` downloads the full filtered range |
| `GET /api/v1/metrics` | `quantity` (`net`, `demand`, `wind`, `solar`), `horizon` (1–24), `cohort` (`delivered`, `selected_only`), `month` (`all` or saved month); includes saved metrics, separately derived MAPE, histogram and labelled display sample |
| `GET /api/v1/audit` | Same historical horizon/cohort/month filters; reason counts and paginated target exclusions; `format=csv` returns all filtered exclusions |
| `GET /api/v1/models` | `component` and `horizon`; native selected-tree gain, selected iteration range and saved metadata |
| `GET /api/v1/logs` | Last 200 actual current-session HTTP events |

Invalid requests return 422. Unavailable artifacts return 503. Unknown API endpoints return JSON 404 rather than the SPA shell. The demo-only synthetic input requires `case: "blocked"` and explicit `demo: true`; it modifies only an in-memory copy. The API never writes measurements or model files.

## Complete sample run and verification

```bash
.venv/bin/python -m pip install -r requirements-command-center-dev.txt
.venv/bin/python scripts/run_command_center_demo.py
npm run --prefix frontend test
npm run --prefix frontend build
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests .venv/bin/python -m unittest \
  test_command_api test_demand_model test_demand_pipeline test_grid_prediction \
  test_grid_evaluation test_grid_curve test_dashboard_service test_dashboard_app
.venv/bin/python src/investigations/verify_grid_curve.py
```

The headless sample creates a fresh `reports/demos/command_center_v1/run-*` directory containing normal/fallback/blocked API responses, CLI parity checks, real metrics/MAPE, native model information, a historical quality CSV, an audit, a model-outage fixture check and hashed receipt. It does not require the HTTP server to be running and never overwrites another run.

For real-Chrome checks and screenshots, start the command center first:

```bash
npm install --prefix /tmp/gridcast-dashboard-browser playwright@1.64.1
node scripts/check_command_center_browser.cjs
```

The optional driver is isolated from project runtime dependencies. Chrome defaults to the standard macOS executable. Override `GRIDCAST_CHROME` or `GRIDCAST_PLAYWRIGHT` if needed. Each check writes a new `browser-*` folder; outage screenshots are explicitly simulated test fixtures, not claims that the real models are missing. Browser checks cover all routes, map interaction, synchronized charts, downloads, filters, retained forecasts, pending state, input failures, API/model outages, navigation, mobile layout, reduced motion and WebGL fallback.

See **`reports/demos/command_center_v1/verification.md`** for completed checks, sample output, screenshots, exact tested versions and protected-file evidence. Native model, data, training and existing dashboard artifacts remain unchanged; no commit or push is included.

Font licenses are bundled under `frontend/public/licenses`. Map source and attribution are recorded in `frontend/public/map-attribution.txt`.
