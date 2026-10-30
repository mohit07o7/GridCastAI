# Milestone 1: Demand Forecasting Baseline & XGBoost Model Evaluation

**Project**: GridCastAI — AI-Based Renewable Energy Demand Forecasting & Grid Optimization  
**Date**: October 2026  
**Status**: Completed  
**Artifact Directory**: `reports/milestones/demand_forecasting_milestone.md`  

---

## 1. Executive Summary

This milestone establishes the 24-hour-ahead electricity demand forecasting benchmark for the GridCastAI project. Using hourly grid observation data across India (January 2024 to June 2025, 12,984 observations), we formulated, evaluated, and compared tabular baseline models against an optimized XGBoost gradient-boosted regression tree model. 

The primary model achieves **2.95% MAPE** and **5,969.71 MW MAE** on the out-of-sample test split, outperforming the Seasonal-Naive benchmark (3.33% MAPE, 6,659.14 MW MAE) and the constant Training Mean benchmark (7.07% MAPE).

---

## 2. Dataset and Chronological Data Split

To prevent temporal data leakage, standard random k-fold cross-validation is strictly avoided. The time series is partitioned chronologically into three non-overlapping windows:

- **Total Observations**: 12,984 hourly timestamps (2024-01-01 00:00:00 to 2025-06-24 23:00:00)
- **Training Set (70%)**: First 9,088 observations (2024-01-01 to 2025-01-13)
  - Used for parameter estimation and feature target encoding
- **Validation Set (15%)**: Next 1,948 observations (2025-01-14 to 2025-04-05)
  - Used for early stopping (patience = 30) and anomaly sensitivity experiments
- **Test Set (15%)**: Final 1,948 observations (2025-04-05 to 2025-06-24)
  - Held-out unseen period used solely for final model performance reporting

---

## 3. Feature Engineering & Target Formulation

- **Forecast Target**: Electricity Demand 24 hours in the future (`Demand(t + 24)`).
- **Lag Features**: Demand at `t - 1`, `t - 24` (daily seasonality), `t - 48` (2-day lag), and `t - 168` (weekly seasonality).
- **Rolling Aggregations**: 24-hour and 168-hour rolling means shifted by 1 hour (ensuring the current observation hour is excluded from the window to eliminate lookahead bias).
- **Calendar Signals**: Hour of day (0–23), Day of week (0–6), and Month (1–12).
- **Integrity Constraints**: Original recorded observations remain unmodified; anomalies are explicitly flagged rather than silently deleted or interpolated.

---

## 4. Empirical Evaluation & Model Comparison

### Test Set Performance (24-Hour Ahead Demand)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Improvement vs Seasonal Naive |
| :--- | :--- | :--- | :--- | :--- |
| **Training Mean Baseline** | 14,880.04 | 18,161.58 | 7.07% | - |
| **Seasonal Naive Baseline** | 6,659.14 | 9,060.72 | 3.33% | Baseline |
| **XGBoost Regressor** | **5,969.71** | **7,668.96** | **2.95%** | **+10.35% MAE / +15.36% RMSE** |

### Top Predictor Importance (XGBoost Split/Gain)
1. `demand_lag_24`: Primary daily seasonal anchor
2. `demand_lag_168`: Weekly recurrence pattern
3. `hour`: Intraday load profile
4. `demand_mean_24h`: Recent baseline consumption level
5. `day_of_week`: Weekend vs weekday consumption divergence

---

## 5. Anomaly Impact & Sensitivity Findings

During validation auditing, an abrupt anomaly was flagged on **April 4, 2025 at 16:00:00**, where demand recorded 97,065.28 MW between 203,399.28 MW at 15:00 and 200,036.27 MW at 17:00 (a >50% temporary dip).

Sensitivity analysis (`reports/experiments/anomaly_sensitivity_validation.csv`):
- **Full Validation Set**: MAE = 4,707.86 MW, RMSE = 6,968.61 MW, MAPE = 2.49%
- **Excluding Flagged Anomaly Target**: MAE = 4,655.40 MW, RMSE = 6,542.32 MW, MAPE = 2.43%
- **Impact**: While this isolated point increases RMSE by 426.29 MW on validation due to quadratic penalty, XGBoost's robust split criterion prevented model overfitting to the outlier.

---

## 6. Key Limitations & Next Milestones

1. **Exogenous Weather Variables**: Currently, weather variables (temperature, solar irradiance, wind speed) are not bundled in the national grid hourly sheet. Integrating ERA5/IMD reanalysis weather data will further enhance peak summer cooling load forecasts.
2. **Dedicated Renewable Generation Models**: Wind generation exhibits long run repetitions (up to 87 consecutive hours) in source recordings during specific periods; separate diagnostic isolation is required before training solar and wind forecasting heads.
3. **Decision Engine**: Next phase connects demand forecasts with renewable forecasts to compute real-time net grid requirement (`Grid Requirement = Demand - (Solar + Wind)`).
