# Streamlit interface retirement verification

The React command center remains the supported local interface. Shared code was moved before removing the old interface:

- `src/dashboard/service.py` → `src/services/grid_service.py` (forecasting/report/export functions unchanged).
- `src/dashboard/runtime.py` → `src/services/runtime.py` (guard error wording updated).
- `tests/test_dashboard_service.py` → `tests/test_grid_service.py` (shared regressions retained).

API imports now use the neutral modules. Removed the Streamlit app/package marker, Streamlit launcher, old browser/demo scripts, interface-specific application tests and `requirements-dashboard.txt`. The installed environment was not bulk-uninstalled: active command-center dependencies and numerical pins are unchanged. No active source, script, test or requirements file imports the retired dashboard package or Streamlit. Historical reports retain their original paths, hashes and version records.

## Checks performed

- 47 Python regression tests passed, including all retained shared-service tests.
- Fresh-process API test blocks `streamlit`, `dashboard` and `plotly` imports; history and a full 24-hour forecast succeed, with training guards still active.
- Five frontend tests passed; strict TypeScript checking and production build passed.
- Launcher preflight loaded reports, history and models successfully.
- Real Chrome smoke check on a fresh server: normal forecast rendered; all eight routes returned successfully with no page exceptions.
- 455 existing model, measurement, report and forecasting-source files matched their pre-change SHA-256 hashes.
- Git whitespace check passed. No retraining, battery execution, commit or push was performed.

The full regression command is maintained in `docs/command_center.md`. Restart an already-running server to load the relocated modules.
