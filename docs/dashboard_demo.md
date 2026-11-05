# Retired Streamlit demonstration

The Streamlit interface, launcher, interface tests and dependency manifest have been removed. Use the [React command center](command_center.md) and `./scripts/start_command_center.sh` instead.

Shared forecasting, report validation and export helpers now live in `src/services/grid_service.py`; runtime guards live in `src/services/runtime.py`. The React API uses these modules directly. Shared service tests remain in `tests/test_grid_service.py`.

The original demonstration reports and screenshots under `reports/demos/dashboard_v1/` remain as historical evidence. Their recorded commands, paths, dependency versions and hashes describe the retired implementation, not the current startup procedure.
