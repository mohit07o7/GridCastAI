# Dashboard delivery verification

Implemented and verified in Desktop/GridCastAI on November 5, 2026. The dashboard is local-only at http://127.0.0.1:8501. Optimization remains **Deferred**; no battery schedules or savings are presented.

- **43 forecasting/dashboard regression tests passed**, including six service tests and three Streamlit application tests.
- **Nine real-Chrome checks passed:** aligned 24-point charts, byte-identical CSV/JSON downloads, saved historical metric filters, separate quality audit, deferred optimization, fallback/actual overlays, blocked-component downloads, narrow layout and absence of browser exceptions.
- Native verification reproduced all **359,352** candidate predictions across **4,991** origins with **0 MW** maximum error. Required tolerance: 0.000001 MW. Existing demand-bundle and single-target command verification also passed.
- Headless startup/demo disable training and battery entry points. The launcher was tested both from the project root and /tmp.
- All **321 pre-existing protected files** remain byte-for-byte unchanged. Original runtime dependency pins remain satisfied; reviewed dashboard packages are pinned separately. `pip check` passed. No commit or push occurred.

## Start and reproduce

```bash
cd /Users/mohityadav/Desktop/GridCastAI
./scripts/start_dashboard.sh
.venv/bin/python scripts/run_dashboard_demo.py
```

One-time setup and the full walkthrough are in [dashboard_demo.md](../../../docs/dashboard_demo.md).

## Complete sample run

[run-1nsmjpb4/walkthrough.md](run-1nsmjpb4/walkthrough.md) contains the walkthrough. The folder includes normal, historical fallback and explicitly synthetic blocked-wind JSON/CSV outputs, a historical accuracy row, an April exclusion audit, checks and a hashed run receipt.

## Browser screenshots

- [Normal forecast](browser-bUgxiy/01-normal.png)
- [Derived quantities and downloads](browser-bUgxiy/01b-normal-details.png)
- [Historical accuracy](browser-bUgxiy/02-accuracy.png)
- [Quality and exclusions](browser-bUgxiy/03-quality.png)
- [Optimization deferred](browser-bUgxiy/04-deferred.png)
- [Historical fallback with actual overlay](browser-bUgxiy/05-fallback.png)
- [Synthetic blocked component](browser-bUgxiy/06-blocked.png)
- [Narrower layout](browser-bUgxiy/07-narrow.png)

`verification.json` records exact tested versions, source/report receipts, code/output hashes and checks. `protected_files.json` records the unchanged pre-existing file hashes. This is a retrospective demonstration using the existing historical dataset, not a newly untouched holdout or a live-data deployment.
