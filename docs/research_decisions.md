# Research Decisions & Anomaly Investigations

**Project**: GridCastAI  
**Path**: `docs/research_decisions.md`  
**Last Updated**: October 2026  

---

## 1. Guiding Research Philosophy

In enterprise data science and academic research, data quality anomalies are opportunities for diagnostic investigation rather than inconveniences to be silently smoothed over. GridCastAI adheres to three core tenets:
1. **Never alter or interpolate raw source observations** without rigorous evidentiary justification.
2. **Flag and isolate anomalies explicitly** via the Anomaly Register (`reports/audits/anomaly_register.csv`).
3. **Conduct empirical sensitivity studies** before deciding how anomalies affect model training and evaluation.

---

## 2. Anomaly A001 — Long Consecutive Repeated Generation Readings

### Observation
Between **2024-06-06 18:00:00** and **2024-06-10 08:00:00** (87 consecutive hourly steps), both `Total Generation (MW)` and `Wind (MW)` report identical values down to the second decimal place:
- `Total Generation (MW)`: exactly `212,887.48 MW`
- `Wind (MW)`: exactly `11,065.84 MW`
Meanwhile, `Demand (MW)` and `Solar (MW)` continued to fluctuate normally during this same 87-hour interval.

### Diagnostic Investigation & Findings
Multiple diagnostic scripts were executed to trace the root cause:
- `src/investigations/inspect_excel_cells.py`: Verified raw cell contents in the Excel workbook.
- `src/investigations/compare_raw_processed.py`: Re-checked merge status and numerical divergence between raw Excel rows and cleaned CSV rows.
- `src/investigations/reproduce_cleaning_check.py`: Independently reproduced the cleaning pipeline logic.
- `src/investigations/audit_repeated_generation_full.py`: Audited the complete 12,984-hour dataset for repeated values.

**Key Finding**:
The repeated values exist verbatim in the source Excel file published by the grid operator. It is not an ETL, rounding, or pandas indexing artifact. Across the full dataset, exactly 10 separate runs of $\ge 6$ consecutive identical hours exist for both Wind and Total Generation (amounting to 250 hours / 1.93% of the dataset), with the 87-hour June 2024 episode being the longest.

### Research Decision
- **Action**: Retain original observations. Do not forward-fill or linearly interpolate.
- **Modeling Consequence**: `Total Generation (MW)` is excluded as a trusted forecasting target until official telemetry validation or correction notes are provided by the grid operator.

---

## 3. Anomaly A002 — April 4, 2025 Demand Plunge

### Observation
On **2025-04-04 at 16:00:00**, `Demand (MW)` plunges to `97,065.28 MW`. 
The surrounding hours record:
- 15:00: `203,399.28 MW`
- **16:00: `97,065.28 MW` (-52.3% drop)**
- 17:00: `200,036.27 MW`

### Sensitivity Experiments
Two distinct counterfactual experiments were conducted:

#### Experiment 1: Target Metric Sensitivity (`reports/experiments/anomaly_sensitivity_validation.csv`)
Validation metrics were evaluated with vs without this single target observation:
- With target: MAE = 4,707.86 MW, RMSE = 6,968.61 MW, MAPE = 2.49%
- Excluding anomalous target: MAE = 4,655.40 MW, RMSE = 6,542.32 MW, MAPE = 2.43%
- **Conclusion**: Due to squaring in RMSE, a single 106,334 MW residual inflates RMSE by 426.29 MW, but the mean absolute error is shifted by only 52.46 MW.

#### Experiment 2: Input Feature Impact Across Full 168-Hour Window (`reports/experiments/anomaly_feature_impact_full_window.csv`)
Because lag features ($L_1, L_{24}, L_{48}, L_{168}$) and rolling averages ($R_{24}, R_{168}$) propagate an anomalous input through time, we trained counterfactual models to measure downstream error propagation. Downstream error spikes were observed when the anomalous hour entered $L_1$ and $L_{24}$, but decayed rapidly thereafter.

### Research Decision
- **Action**: Preserve the true reported observation in the dataset.
- **Implementation**: The feature pipeline adds `known_demand_anomaly = (Timestamp == '2025-04-04 16:00:00')` in `data/processed/demand_model_data.csv`. This enables any model to condition on or filter the record transparently without silent data corruption.

---

## 4. Telemetry Telecommunications & Sensor Drift (Negative Solar)

### Observation
Nighttime solar dispatch recordings contain values between `-0.01 MW` and `-13.78 MW` across 1,124 observations, predominantly between 19:00 and 05:00.

### Technical Assessment
Solar photovoltaic farms consume small amounts of power during darkness to operate telemetry sensors, inverter standby units, and substation transformers. Some SCADA systems record this negative net flow into the farm.

### Research Decision
- Maintain dual columns in `data/processed/hourly_grid_data_validated.csv`:
  - `Solar (MW) reported`: Untouched SCADA telemetry record.
  - `Solar (MW) corrected`: Clipped at zero (`clip(lower=0)`) to represent physical generation available to supply the consumer grid.

---

## 5. Summary of Anomaly Register

| Anomaly ID | Time Window | Columns Affected | Severity | Status | Decision & Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A001** | 2024-06-06 18:00 to 2024-06-10 08:00 | Total Generation; Wind | High | Confirmed Source Telemetry Freezing | Preserved in raw/processed; Total Generation quarantined from primary model target |
| **A002** | 2025-04-04 16:00:00 | Demand; Total Generation | High | Flagged Plunge (>50%) | Preserved in data; explicit boolean flag added; sensitivity audited |
