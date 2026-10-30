# Demand refresh — 31 October 2026

Desktop/GridCastAI is the main project. The previous workspace is retained at Documents/ChatGPT/GridCaseAi-archive-2026-10-31 after the three spot-check files were copied, verified byte-for-byte, and the Desktop model replayed successfully.

All four demand experiments now import common features and purged chronological splits. The original origin-relative feature values/order are preserved. The ablation experiment retains its original feature order (calendar before renewable columns); stochastic column subsampling can therefore produce a small difference from the main model even with the same 13 features.

All four experiments ran first with the shared purge alone and then with quality policy v1. Pre-refactor code/results/figures are in reports/historical/pre_shared_2026_10_31; intermediate shared-purge results are in reports/historical/shared_purge_before_quality_2026_10_31. Only reports/experiments contains the current four demand experiment outputs. The old grid_requirement_evaluation.csv was archived because every demand forecast in it belongs to the earlier unpurged run; it has no current reproducible producer and must be rebuilt in a later grid integration task.

## Refreshed results

Training 22,548 rows; validation 4,962; test 4,603. Nominal test boundaries unchanged. Source measurements unchanged.

| Experiment | Test MAE MW | Test MAPE % |
|---|---:|---:|
| Main demand | 5106.45 | 2.5892 |
| Prior-day persistence | 5416.56 | 2.8114 |
| Demand-only ablation | 4961.88 | 2.5247 |
| Demand + renewables ablation | 5115.99 | 2.5919 |
| Trend experiment | 4724.11 | 2.3996 |

Seven-fold unweighted mean MAE: demand-only 5124.70 MW; demand + renewables 5167.33 MW. Fold variability is recorded in demand_rolling_summary.csv. Trend performed best among these final-test experiments, but selecting it after observing these test scores requires confirmation on future data.

The historical purged main run scored 5312.82 MW MAE / 2.7283% on 4,986 test hours. Because v1 excludes 383 test rows and changes fitting data, that result is not directly comparable with the refreshed score. All current model/baseline comparisons use identical eligible rows.

## Spot-check and verification

Historical Jan 10 target: 201367.78125 MW, matching the original 201367.78 saved CSV. Current policy: excluded due to frozen demand within its input history. The diagnostic model output is 202477.65625 MW; it is not included in headline metrics. Both full-test replays match their saved predictions within 0.00625 MW. Evidence is in demand_spot_check_historical_replay.json and demand_spot_check_current_policy.json. The original demand_spot_check.md/json are retained as historical investigation records, including their original workspace paths.

Commands from the Desktop root:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python src/investigations/reproduce_demand_spot_check.py --historical --retrain
.venv/bin/python src/investigations/reproduce_demand_spot_check.py --retrain
.venv/bin/python -m pip check
git ls-files data
```

Shared-code tests cover future-value invariance, lag/rolling alignment, purge assertions, duplicate rejection, frozen flags, and the exact inclusive quality-window boundary. Current prediction origins and headline metrics were checked against shared eligibility and saved actuals. Python 3.14.3 and curated package pins are documented in docs/environment.md; all stochastic estimator constructors use seed 42. No data files belong in Git.
