# Data Dictionary

**Project**: GridCastAI  
**Path**: `docs/data_dictionary.md`  
**Last Updated**: October 2026  

---

## 1. Cleaned & Validated Dataset Schema
Files: `data/processed/hourly_grid_data_clean.csv`, `data/processed/hourly_grid_data_validated.csv`

| Field Name | Type | Unit | Nullable | Valid Range | Definition & Engineering Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Timestamp` | Datetime (UTC/IST) | - | No | 2024-01-01 to 2025-06-24 | Hourly time step index (continuous 1-hour cadence). |
| `Demand (MW)` | Float | Megawatts | No | [97,065.28, 249,796.65] | Total grid power consumption. Anomaly flagged at 2025-04-04 16:00:00. |
| `Wind (MW)` | Float | Megawatts | No | [319.46, 31,438.35] | Total wind generation dispatched. Contains repeated-value runs in June 2024. |
| `Solar (MW)` | Float | Megawatts | No | [-13.78, 48,930.68] | Raw recorded solar dispatch. Negative values occur during night hours due to telemetry offset/auxiliary load. |
| `Solar (MW) reported` | Float | Megawatts | No | [-13.78, 48,930.68] | Preserved audit copy of original solar measurement (in `validated.csv`). |
| `Solar (MW) corrected` | Float | Megawatts | No | [0.00, 48,930.68] | Physically bounded solar feature (`clip(lower=0)`). Negative night readings set to 0.00 MW. |
| `Total Generation (MW)` | Float | Megawatts | No | [121,570.62, 252,384.85] | Aggregate generation dispatched from all fuel sources. |

---

## 2. Demand Modeling Dataset Schema
File: `data/processed/demand_model_data.csv`

| Field Name | Type | Valid Range | Role | Definition |
| :--- | :--- | :--- | :--- | :--- |
| `Timestamp` | Datetime | 2024-01-01 to 2025-06-24 | Index | Hourly timeline indicator. |
| `Demand (MW)` | Float | [97,065.28, 249,796.65] | Base Series | Current observed power demand. |
| `hour` | Integer | [0, 23] | Feature | Hour of day (diurnal cyclical pattern). |
| `day_of_week` | Integer | [0, 6] | Feature | Day index (0 = Monday, 6 = Sunday; distinguishes weekend load reduction). |
| `month` | Integer | [1, 12] | Feature | Month index (captures seasonal cooling and heating trends). |
| `known_demand_anomaly` | Boolean | True / False | Flag | `True` solely for 2025-04-04 16:00:00; preserves observation while enabling sensitivity audits. |

---

## 3. Dynamically Derived Features (Model Runtime)
Constructed in `src/forecasting/xgboost_demand_forecast.py` and `src/analysis/analyze_anomaly_sensitivity.py`:

| Feature Name | Computation Formula | Purpose |
| :--- | :--- | :--- |
| `Target` | `Demand(t + 24)` | Forecast objective: demand 24 hours into the future. |
| `demand_lag_1` | `Demand(t - 1)` | Captures immediate autocorrelation / momentum. |
| `demand_lag_24` | `Demand(t - 24)` | Daily seasonal baseline at the forecast hour. |
| `demand_lag_48` | `Demand(t - 48)` | Two-day backcheck for trend confirmation. |
| `demand_lag_168` | `Demand(t - 168)` | Weekly seasonal baseline (same hour, same day last week). |
| `demand_mean_24h` | $\frac{1}{24}\sum_{i=1}^{24} \text{Demand}(t - i)$ | Recent daily moving average level (shifted by 1h to avoid lookahead bias). |
| `demand_mean_168h` | $\frac{1}{168}\sum_{i=1}^{168} \text{Demand}(t - i)$ | Weekly moving baseline demand level. |

---

## 4. Derived Decision Metrics
Used in energy consistency checks (`src/analysis/check_energy_consistency.py`):

- **Net Grid Requirement**:
  $$\text{Grid Requirement (MW)} = \text{Demand (MW)} - \text{Wind (MW)} - \text{Solar (MW) corrected}$$
  Represents non-renewable thermal, hydro, and imported generation required to prevent grid frequency collapse.
- **Generation Imbalance / Reserve Margin**:
  $$\text{Generation Margin (MW)} = \text{Total Generation (MW)} - \text{Demand (MW)}$$
