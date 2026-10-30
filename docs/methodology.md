# Research Methodology

**Project**: GridCastAI  
**Path**: `docs/methodology.md`  
**Last Updated**: October 2026  

---

## 1. Data Ingestion & Preprocessing Pipeline

The data engineering pipeline guarantees that raw records are never altered while preparing structurally clean, audit-compliant data for machine learning models.

```
data/raw/January 2024- June 2025.xlsx
                  │
                  ▼  (src/preprocessing/clean_hourly_data.py)
data/processed/hourly_grid_data_clean.csv
                  │
                  ▼  (src/preprocessing/validate_solar.py)
data/processed/hourly_grid_data_validated.csv
                  │
                  ▼  (src/preprocessing/prepare_demand_data.py)
data/processed/demand_model_data.csv
```

### Preprocessing Principles:
1. **Datetime Parsing & Summary Cleansing**: Excel reports contain aggregate rows at the footer. Timestamps are coerced with `pd.to_datetime(..., dayfirst=True, errors='coerce')` and invalid rows are discarded.
2. **Interval Continuity**: The series is verified to have an exact continuous hourly step ($\Delta t = 1\text{ hour}$). Any missing step triggers a pipeline error rather than silent forward-filling.
3. **Audit Preservation (Solar Rule)**: Telemetry records occasionally report small negative solar generation values during night hours (e.g. -0.5 to -13.78 MW) caused by inverter tare consumption or sensor calibration drift. Rather than overwriting the raw column, the pipeline preserves `Solar (MW) reported` and creates a physically constrained `Solar (MW) corrected = clip(lower=0)`.

---

## 2. Feature Engineering & Leakage Prevention

For real-time operational grid dispatch, a model making a 24-hour-ahead forecast at time $t$ has access only to information up to and including hour $t$.

### Feature Formulation:
- **Autoregressive Lags**:
  - $L_1 = \text{Demand}(t - 1)$
  - $L_{24} = \text{Demand}(t - 24)$ (daily recurrence)
  - $L_{48} = \text{Demand}(t - 48)$
  - $L_{168} = \text{Demand}(t - 168)$ (weekly recurrence)
- **Moving Window Aggregations**:
  - `demand_mean_24h` and `demand_mean_168h` are computed using `.shift(1).rolling(w).mean()`. Shifting by 1 guarantees that current-hour and future-hour measurements are strictly barred from the rolling statistic.
- **Calendar Encodings**:
  - `hour`: Captures daily human activity and industrial schedule peaks.
  - `day_of_week`: Differentiates weekday base load from weekend reductions.
  - `month`: Captures seasonal weather variations across summer peak cooling and winter heating.

---

## 3. Chronological Evaluation Splits

Standard $k$-fold cross-validation is invalid for autocorrelated time series because random shuffling leaks future information into past predictions. 

GridCastAI partitions the 12,984 observations strictly along the timeline:

```
[────────── 70% Train (9,088 hrs) ──────────][── 15% Val (1,948 hrs) ──][── 15% Test (1,948 hrs) ──]
2024-01-01                             2025-01-13                 2025-04-05                    2025-06-24
```

- **Training Set (70%)**: Used solely to fit model parameters and learn tree split structures.
- **Validation Set (15%)**: Used for hyperparameter tuning and early stopping regularization (`early_stopping_rounds=30`).
- **Test Set (15%)**: An entirely untouched holdout window representing true future operational conditions.

---

## 4. Benchmark Models & Performance Metrics

To prove that complex models provide genuine value, every machine learning model is evaluated against established domain baselines:

1. **Training Mean Benchmark**:
   $$\hat{y}_{t+24} = \bar{y}_{\text{train}}$$
2. **Seasonal-Naive Benchmark (24-Hour Persistence)**:
   $$\hat{y}_{t+24} = y_{t}$$
   *(Uses the observed demand from the identical hour on the previous day)*
3. **XGBoost Regressor**:
   Gradient-boosted decision trees optimizing squared error loss with shrinkage learning rate $\eta = 0.05$.

### Error Metrics:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
