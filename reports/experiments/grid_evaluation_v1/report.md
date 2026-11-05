# Combined grid evaluation v1

This is a retrospective evaluation. These historical periods have already informed model and quality-policy development.

## What was compared

Saved forecasts were joined by both origin and target timestamp. Command replay used the same saved models, history-only quality checks, and labelled fallbacks. Both methods and persistence always share the same scoring rows within each cohort.

Net demand = demand − wind − solar. Renewable contribution is wind + solar, with its share expressed relative to demand. Net demand remains signed; this is a residual supply requirement, not a dispatch or battery optimization.

| Cohort | Scored hours | Forecast MAE (MW) | Persistence MAE (MW) | MAE reduction |
|---|---:|---:|---:|---:|
| saved_benchmark | 4363 | 5,624.03 | 5,897.52 | 4.64% |
| replay_selected_only | 4282 | 5,592.48 | 6,026.85 | 7.21% |
| replay_delivered | 4890 | 5,849.98 | 6,112.89 | 4.30% |

## What the result means

- saved_benchmark: forecast MAE is 273.49 MW lower than persistence. The approximate paired day-bootstrap 95% interval for MAE improvement is [-193.84, 755.48] MW and includes zero; the improvement is inconclusive under this resampling check. It improves in 7/8 scored months and worsens in 1; the benefit is not consistent across months.
- replay_selected_only: forecast MAE is 434.37 MW lower than persistence. The approximate paired day-bootstrap 95% interval for MAE improvement is [-16.39, 857.41] MW and includes zero; the improvement is inconclusive under this resampling check. It improves in 7/8 scored months and worsens in 1; the benefit is not consistent across months.
- replay_delivered: forecast MAE is 262.91 MW lower than persistence. The approximate paired day-bootstrap 95% interval for MAE improvement is [-205.01, 675.28] MW and includes zero; the improvement is inconclusive under this resampling check. It improves in 7/8 scored months and worsens in 1; the benefit is not consistent across months.

## Coverage and exclusions

- Nominal test window: 4986 origins, 2024-11-28 06:00:00 through 2025-06-23 23:00:00.
- Saved common rows: 4363; missing-pattern counts are {'[]': 4363, '["demand"]': 383, '["solar"]': 240}.
- Replay trusted targets: 4919; target exclusions: 67.
- Delivered forecasts scored: 4890; scored rows using a fallback: 608.
- Any blocked component: 53 origins, including 29 trusted-target origins.

The target policy excludes invalid/nonpositive-demand actuals, all-zero records, the April 4 demand anomaly, frozen demand/wind targets, positive solar flatlines and recorded renewable inconsistencies. Full-series retrospective flags are attached only after prediction generation. Other unclassified sharp changes are not automatically removed from target scoring.

Selected-only and delivered replay scores use different row sets; do not rank them by their headline MAE alone. Coverage and monthly results show which hours each serves. The saved benchmark retains its inherited experiment exclusions. The demand prediction CSV has up to 0.00625 MW formatting differences from full-precision model outputs; the replay comparison allows 0.02 MW.

All excluded origins and reasons are in alignment_audit.csv and replay_rows.csv. Missing component actuals are excluded from target scoring; a blocked forecast remains missing, never silently zero. Undefined renewable-share percentages are recorded and paired share metrics use a common valid-share mask.

Metrics use signed error = forecast − actual. Positive MAE improvement means the combined forecast is better. WAPE uses the sum of absolute actual net demand, avoiding per-hour division near zero. Bootstrap resamples target calendar-day blocks (1,000 draws, seed 42); it is an exploratory interval, not proof of future or cross-region generalization.

## Reproduce

```bash
.venv/bin/python src/analysis/evaluate_grid_forecasts.py
.venv/bin/python src/investigations/verify_grid_evaluation.py
```

No training, source measurement changes or model replacement occurs. summary.json records artifact/data/code hashes, policies, exclusion counts and paired metrics. monthly_metrics.csv records month-by-month scores and coverage. verification.json contains independent exported-row checks.
