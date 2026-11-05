# GridCastAI cinematic command center — verification

Delivered in Desktop/GridCastAI on November 5, 2026. Open http://127.0.0.1:8502 or start with `./scripts/start_command_center.sh`.

The new application uses React, strict TypeScript, Vite, Tailwind, Framer Motion, ECharts, MapLibre, Lucide, React Router, Zustand, TanStack Query and Axios. It runs independently from the preserved Streamlit app and connects to the real verified Python service through a local read-only API.

## Passed checks

- **49 Python tests** covering the API and all existing forecasting/dashboard regressions.
- **5 frontend tests** for signed arithmetic, missing/undefined values, source-clock arithmetic, separate actual overlays and CSV escaping.
- **20 real-Chrome checks** across all eight routes, map pan/zoom/reset/selection, synchronized target-time tooltips, historical filters, selected-tree gain, downloads, draft retention, actual pending state, fallbacks, blocked inputs, API/model-outage fixtures, full-resolution historical export, mobile/keyboard navigation, reduced motion and static-map fallback.
- Strict TypeScript check and production build passed. Native verification reproduced all **359,352** reference predictions at **0 MW maximum error**, within the required 0.000001 MW tolerance. Existing demand-bundle and single-target prediction verification also passed.
- The complete headless API sample matches the existing CLI **exactly**. MAPE uses a separately labelled identical nonzero-actual mask; WAPE remains the saved metric. Feature gain uses only selected prediction trees.
- All **391 pre-existing protected files** remain byte-for-byte unchanged, including existing models, measurements, training code, research outputs and Streamlit implementation. Existing numerical pins remain satisfied. `pip check` and `git diff --check` passed. No commit or push occurred.
- Startup was tested from the project root and /tmp. It installs nothing, trains nothing and invokes no battery solver. The API serves the production React build on loopback only.

## Complete sample and screenshots

The complete API sample is [run-hby16c79/run.json](run-hby16c79/run.json). It includes real normal/fallback forecasts, an explicitly synthetic blocked-wind example, metrics/MAPE, native feature gain, a source-history CSV, target exclusions and an artifact-outage fixture check.

| View | Screenshot |
|---|---|
| Command Center | [Open](browser-pr2bIG/01-command-center.png) |
| Grid Demand Intelligence | [Open](browser-pr2bIG/page-demand.png) |
| Renewable Energy Monitoring | [Open](browser-pr2bIG/page-renewables.png) |
| AI Forecasting Lab | [Open](browser-pr2bIG/page-forecast-lab.png) |
| Grid Requirement Analysis | [Open](browser-pr2bIG/page-grid-requirement.png) |
| Model Intelligence | [Open](browser-pr2bIG/page-models.png) |
| Historical Data Explorer | [Open](browser-pr2bIG/page-history.png) |
| System Operations | [Open](browser-pr2bIG/page-operations.png) |
| Historical fallback + reported actuals | [Open](browser-pr2bIG/09-historical-fallback.png) |
| Synthetic blocked wind | [Open](browser-pr2bIG/10-synthetic-blocked.png) |
| Browser-simulated API outage | [Open](browser-pr2bIG/11-api-outage-fixture.png) |
| Browser-simulated model outage | [Open](browser-pr2bIG/12-model-outage-fixture.png) |
| Narrower layout | [Open](browser-pr2bIG/13-narrow.png) |
| Mobile navigation/layout | [Open](browser-pr2bIG/14-mobile.png) |
| Static geographic fallback | [Open](browser-pr2bIG/15-static-map-fallback.png) |
| Reduced-motion view | [Open](browser-pr2bIG/16-reduced-motion.png) |

## Interpretation and reproducibility

Coverage is **India national aggregate**. The map is an attributed Natural Earth reference, not regional electrical telemetry or transmission topology. Forecast dates use the original timezone-naive source clock. Historical overlays and quality flags never enter future features. Missing values remain missing; net demand stays signed. Historical results are retrospective, not a newly untouched holdout or evidence of live operation. Solar uses the existing corrected nonnegative `solar_mw` target.

**Optimization remains Deferred.** No battery schedule, optimized dispatch or simulated savings is displayed.

See [the setup, architecture, API and walkthrough guide](../../../docs/command_center.md). `verification.json` records exact tested versions, code/build/demo hashes and checks; `protected_files.json` records the original unchanged-file hashes. Browser-failure fixtures are clearly labelled tests and do not alter actual model artifacts or source data.
