# Four-Year Dataset: Modeling Readiness Audit

**Audited:** `data/processed/full_hourly_stacked.csv`  
**Audit date:** 21 October 2026  
**Decision:** Structurally suitable for controlled modeling experiments **after** quality flags and split rules are added. It is not yet a clean, unqualified training target.

**Existing four-year run:** `src/forecasting/xgboost_demand_forecast.py` already trains on `data/processed/demand_model_data.csv`, which begins in September 2021 and has 33,432 rows. Its 4,986 saved test predictions reproduce **5,268.38 MW MAE / 2.71% MAPE** for XGBoost and **5,672.32 MW MAE / 2.96% MAPE** for true prior-day demand persistence. This is a separate run from the older 2.95% MAPE demand milestone. It should be rerun or reassessed after the quality and split rules below are applied; the existing result is not proof that the stack is fully validated.

## Verified structure

The stack has **33,432 rows**, from **2021-09-01 00:00** through **2025-06-24 23:00** inclusive. There are **zero duplicate timestamps**, **zero non-hourly intervals**, and **zero missing values** in demand, wind, raw/corrected solar, combined solar and wind, and total generation. Every hour has the expected number of source rows according to the saved `rows_per_hour` metadata. `reports/audits/full_dataset_model_readiness.json` and `full_dataset_era_summary.csv` contain the reproducible counts.

| Source era | Dates | Hours | Source files | Hourly construction |
| --- | --- | ---: | ---: | --- |
| Five-minute | Sep 2021–Sep 2022 | 9,480 | 13 | Mean of 12 source rows per hour |
| Ten-second | Oct 2022–Dec 2023 | 10,968 | 15 | Mean of 360 source rows per hour |
| Hourly | Jan 2024–Jun 2025 | 12,984 | 1 | Passed through from the hourly workbook |

The 29 files are marked `OK` in `reports/audits/resample_run_log.csv`. The counts establish timestamp and row coverage, but do not prove that every observed signal value is valid or that the three source eras are semantically identical.

## Quality flags that affect training

| Finding | Evidence | Modeling implication |
| --- | --- | --- |
| All-zero records | **12 hours** with demand, wind, solar, and total generation all exactly zero in four legacy workbooks. All have a full expected source-row count. The raw workbooks confirm that every one of their 12 five-minute source rows has zero in measurement columns F–K. See `full_dataset_all_zero_hours.csv` and `legacy_source_anomaly_verification.json`. | These are source-recorded zeros, consistent with an outage or placeholder, not resampling gaps. Their operational cause is unverified. Flag them and avoid training or scoring on affected targets without an explicit policy. |
| Combined renewable inconsistency | **Two hours**, 2022-10-01 12:00 and 13:00, where recorded solar-plus-wind exceeds component sum by **21,363 MW** and **30,317 MW**. The raw workbook shows zero wind and solar components while the combined channel remains positive for **180** and **260** ten-second rows respectively, from 12:30:00 to 13:43:10. See `full_dataset_solar_wind_residuals.csv` and `legacy_source_anomaly_verification.json`. | This is a source-channel dropout, not a resampling artifact. Do not use the component or combined signals from these hours as unquestioned targets or features; keep a flag and resolve the channel choice explicitly. |
| Frozen values | **11 runs of at least six hours** in both wind and total generation; 257 hours in each set of runs. Longest is **87 hours** from 2024-06-06 18:00. Demand has three runs of at least six hours, longest 11 hours from 2025-01-08 20:00. | Preserve records and add masks for sensitivity tests. The earlier A001 audit covers the 2024–25 subset; the 11-run count here covers the full four-year stack. |
| Positive solar flatline | **48 consecutive hours** from 2025-03-06 12:00 through 2025-03-08 11:00 are exactly **64,300.61 MW**, including overnight hours. The original hourly workbook contains the same repeated value. See `full_dataset_positive_solar_flatline.csv` and `legacy_source_anomaly_verification.json`. | Flag this source artifact in solar targets and feature windows. It materially distorts solar evaluation if scored as genuine production. |
| Sharp demand changes | **23** hour-to-hour changes above 40% when the previous hour is positive, including the documented 2025-04-04 16:00 plunge. See `full_dataset_demand_jumps.csv`. | Review at source and avoid automatic interpolation. Zero-valued outages explain some jumps; the remainder need classification. |
| Solar measurement correction | **2,784 hourly raw solar means are negative**; corrected solar has no negative values. | Model corrected nonnegative solar; preserve raw measurements for audit. The legacy pipeline clips the derived `Solar` signal before hourly averaging, while preserving the raw telemetry signal separately. |

Demand, wind, and total generation each have 12 nonpositive values, all from the all-zero records. The two renewable residuals have magnitude far above the normal 1 MW tolerance. These findings mean a zero-missing-value check alone overstates readiness.

## Source comparability and provenance

The source changed cadence twice, at **2022-10-01** and **2024-01-01**, and the newer workbook uses a different sheet and field names. The first timestamps across both boundaries have plausible levels, but this is only a spot check. `src/preprocessing/resample_to_hourly.py` uses an arithmetic mean of MW readings for legacy hours and passes through the newer hourly values. Before claiming a single homogeneous target, verify in raw source documentation that the legacy channels and newer fields have the same measurement definitions, geographic coverage, units, timezone, and hourly timestamp convention. The current audit does not establish that equivalence.

The targeted source checks are reproducible with `python src/investigations/verify_legacy_source_anomalies.py --raw-dir /Users/mohityadav/Desktop/GridCastAI/data/raw`. They cover the 12 zero hours, the October 2022 component dropout, and the 48-hour solar flatline; they do not re-audit every source row.

The audit was prepared in `/Users/mohityadav/Documents/ChatGPT/GridCaseAi/` using a byte-identical copy of the processed stack from `/Users/mohityadav/Desktop/GridCastAI/`; SHA-256 for both stack copies is `717ad938f1ec651acac450e5c009aef18bc8cd5fcaebfc7c17d12f7e49f65afc`. The 1.4 GB raw workbooks remain in the Desktop project and were not duplicated in the audit workspace.

## Criteria before four-year model results are reported

1. Add a quality flag table keyed by timestamp for all-zero hours, renewable residuals, frozen runs (including solar), and reviewed sharp jumps. Keep the original values unchanged. Record whether flagged targets and their lag windows are excluded, masked, or scored separately.
2. Define train, validation, and test by available information at the forecast origin. Ensure a 24-hour target is already observable before its row is eligible for fitting or tuning at the next split boundary.
3. Correct the seasonal baseline to use the actual at **target minus 24 hours**. The current renewable baseline scripts use target minus 48 hours; see `reports/milestones/renewable_forecasting_milestone.md`.
4. Use an unchanged final holdout and report performance both across all eligible hours and after explicit anomaly exclusions. Compare years and source eras in diagnostics so an apparent model gain is not driven by a source change.
5. Validate the two major source transitions and choose a documented treatment for the raw-confirmed October 2022 component dropout before using the combined renewable field in optimization.

Run `python src/auditing/audit_full_model_readiness.py` from the project root to refresh the audit JSON, era CSV, anomaly CSVs, and renewable baseline recheck. The script reads only processed data and saved prediction files; it does not retrain or modify measurements.
