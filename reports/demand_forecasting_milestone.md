# GridCastAI — Demand Forecasting Milestone

## Objective
Forecast electricity demand 24 hours ahead using hourly observations from January 1, 2024, to June 24, 2025.

## Dataset preparation
- 12,984 hourly observations in the validated dataset.
- No invalid timestamps, duplicate timestamps, missing measurement values, or gaps in the hourly sequence.
- One isolated demand anomaly was flagged at April 4, 2025, 16:00.
- Original source data was preserved.

## Evaluation design
- Chronological train/validation/test split.
- Approximate split: 70% / 15% / 15%.
- XGBoost used historical demand lags, shifted rolling means, and calendar features.
- The final comparison evaluated both models on the same 1,919 test targets.

## Results

| Metric | XGBoost | Seasonal Naïve |
|---|---:|---:|
| MAE (MW) | 5,969.71 | 6,659.14 |
| RMSE (MW) | 7,668.96 | 9,060.72 |
| MAPE (%) | 2.95 | 3.33 |

XGBoost reduced MAE by 10.35%, RMSE by 15.36%, and MAPE by 11.41% relative to the seasonal-naïve benchmark.

## Interpretation
XGBoost outperformed the seasonal-naïve benchmark on the evaluated test period. Historical demand, particularly the 24-hour and 1-hour lags, accounted for the largest feature-importance values.

## Limitations
- The dataset contains suspicious repeated values in wind and total generation.
- Negative reported solar observations require further treatment analysis.
- Demand anomaly sensitivity has not yet been quantified.
- Results represent one chronological test period and do not alone establish generalization to future conditions.
- Statistical significance and forecast performance across alternative time periods have not yet been assessed.

## Next steps
1. Test demand-model sensitivity to the flagged anomaly.
2. Establish a consistent forecasting evaluation protocol.
3. Develop and evaluate wind and solar forecasting models.
4. Derive grid requirement only after documenting the measurement assumptions.
