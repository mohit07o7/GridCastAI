# GridCastAI — AI-Based Renewable Energy Demand Forecasting & Grid Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20research-emerald.svg)]()
[![Model](https://img.shields.io/badge/primary%20model-XGBoost%20Regressor-orange.svg)]()
[![Saved four-year demand MAPE](https://img.shields.io/badge/saved%204--year%20demand%20MAPE-2.71%25-green.svg)]()

GridCastAI is an end-to-end data science and grid optimization research project designed to forecast hourly electricity demand and renewable generation across India. The project emphasizes **telemetry data auditing**, **forecast-origin feature checks**, **systematic anomaly sensitivity analysis**, and **decision-support metrics**.

**Audit workspace:** The 21 October 2026 review used a copy of the processed data and project files under `/Users/mohityadav/Documents/ChatGPT/GridCaseAi/`. The 1.4 GB raw workbooks remain under `/Users/mohityadav/Desktop/GridCastAI/data/raw/` and were not duplicated in the audit workspace.

---

## 1. Project Directory Layout

```
GridCastAI/
├── data/
│   ├── raw/                 # Original, immutable Excel workbooks (.xlsx)
│   ├── interim/             # Temporary intermediate transformed datasets
│   └── processed/           # Cleaned, validated, model-ready CSVs
├── src/
│   ├── auditing/            # Workbook format validation and anomaly registers
│   ├── preprocessing/       # Ingestion, cleaning, solar calibration, feature prep
│   ├── analysis/            # EDA, energy consistency, and anomaly sensitivity
│   ├── forecasting/         # Baseline benchmarks and XGBoost forecasting
│   └── investigations/      # Diagnostic probes and source workbook checks
├── reports/
│   ├── audits/              # Data quality summaries and anomaly register CSVs
│   ├── experiments/         # Model metrics, baseline comparisons, sensitivity outputs
│   └── milestones/          # Formally documented research progress and conclusions
├── outputs/
│   ├── figures/             # High-resolution exploratory and profile plots (.png)
│   └── predictions/         # Model predictions and deliverable forecasts
├── models/                  # Serialized model artifacts, hyperparameter configs
├── notebooks/               # Interactive exploration and experimentation
├── docs/                    # Methodology, data dictionary, sources, research decisions
├── README.md                # Project overview and run instructions
├── requirements.txt         # Pinned project dependencies
└── .gitignore               # Ignored local files, environments, and caches
```

---

## 2. Model Performance Benchmarks

The earlier demand milestone used the 2024–25 dataset with a chronological **15% holdout** (1,948 future hours from April to June 2025):

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Baseline Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **Training Mean** | 14,880.04 | 18,161.58 | 7.07% | Constant historical mean |
| **Seasonal Naive (24h Persistence)** | 6,659.14 | 9,060.72 | 3.33% | Previous day same-hour demand |
| **XGBoost Regressor** | **5,969.71** | **7,668.96** | **2.95%** | **10.35% MAE reduction vs Seasonal Naive** |

A later **four-year demand experiment already exists** in `reports/experiments/xgboost_demand_metrics.csv`. Its saved 4,986 test predictions (target dates 2024-11-29 through 2025-06-24) reproduce **5,268.38 MW MAE / 2.71% MAPE** for XGBoost and **5,672.32 MW MAE / 2.96% MAPE** for a correctly aligned prior-day baseline. These are separate experiments; the four-year result still needs the quality-flag and split review described in the [readiness audit](reports/audits/full_dataset_model_readiness.md).

The saved renewable experiments need a corrected benchmark. Their script's “seasonal naive” uses a value 48 hours before the target, although the label says previous day. On the same saved test timestamps, true previous-day persistence scores **2,322.6 MW MAE for wind** and **1,131.4 MW MAE for solar**, both better than the saved XGBoost models. See [Milestone 2](reports/milestones/renewable_forecasting_milestone.md) and the [four-year modeling readiness audit](reports/audits/full_dataset_model_readiness.md) before citing renewable model gains.

The new [four-year renewable v3 experiment](reports/milestones/renewable_v3_full_history.md) corrects the baseline and split timing. On its eligible test hours, wind residual XGBoost achieves **1,894.9 MW MAE** versus **1,956.3 MW** for prior-day persistence. Solar XGBoost does not reliably improve MAE; prior-day persistence remains selected (**1,154.6 MW MAE** on the solar-eligible test rows). The v2 and v3 test windows differ, so their metrics should not be compared directly.

---

## 3. Workflow Execution Guide

All scripts use robust project-root relative resolution and can be executed either from the repository root or from within their respective directories.

### Step 1: Environment Setup
```bash
# Activate the existing virtual environment
source .venv/bin/activate

# Or install dependencies from requirements.txt
pip install -r requirements.txt
```

### Step 2: Ingestion & Data Preparation
```bash
# 1. Clean the raw Excel workbook and verify hourly continuity
python src/preprocessing/clean_hourly_data.py

# 2. Preserve reported solar readings while creating zero-clipped solar values
python src/preprocessing/validate_solar.py

# 3. Generate modeling features and chronological anomaly flags
python src/preprocessing/prepare_demand_data.py
```

### Step 3: Dataset Audits & Anomaly Verification
```bash
# Audit raw workbook formats and missing values
python src/auditing/validate_all_datasets.py

# Build the consolidated anomaly register
python src/auditing/create_anomaly_register.py

# Check model data readiness
python src/auditing/validate_model_data.py

# Audit the four-year stack and recheck renewable baselines
python src/auditing/audit_full_model_readiness.py
```

### Step 4: Exploratory Analysis & Anomaly Diagnostics
```bash
# Generate EDA demand and renewable profile plots
python src/analysis/eda_grid_data.py

# Analyze demand vs renewable energy balance
python src/analysis/check_energy_consistency.py

# Audit consecutive repeated runs in Wind and Total Generation
python src/investigations/audit_repeated_generation_full.py

# Test target sensitivity to the April 4, 2025 anomaly
python src/analysis/analyze_anomaly_sensitivity.py

# Test 168-hour rolling feature impact of the anomaly
python src/analysis/analyze_anomaly_feature_impact.py
```

### Step 5: Forecasting Experiments
```bash
# Evaluate baseline benchmarks (Seasonal Naive vs Mean)
python src/forecasting/compare_demand_baselines.py

# Train and evaluate the XGBoost demand forecasting model
python src/forecasting/xgboost_demand_forecast.py

# Four-year wind and solar residual experiments with quality flags
python src/forecasting/renewable_v3_full_history.py
```

---

## 4. Key Research Discoveries & Decision Log

1. **Source Repeated Telemetry (Anomaly A001)**:
   - Wind and Total Generation contain 10 distinct episodes of $\ge 6$ consecutive identical readings across the dataset, led by an 87-hour frozen block in June 2024.
   - Traced via `src/investigations/` to confirm that the repeated values originate in the official source workbook from the grid operator. Total Generation is quarantined from primary model targets.
2. **April 4, 2025 Demand Plunge (Anomaly A002)**:
   - Demand fell abruptly from 203,399 MW to 97,065 MW at 16:00 before returning to 200,036 MW at 17:00.
   - Rather than silently imputing, the observation is preserved with an explicit boolean indicator (`known_demand_anomaly`), enabling auditable sensitivity comparisons.
3. **Solar Nighttime Inversion**:
   - Small negative solar values (sensor zero-drift and auxiliary inverter load) are preserved in `Solar (MW) reported` and corrected via `Solar (MW) corrected = clip(lower=0)`.

---

## 5. Documentation Index

- [docs/data_sources.md](docs/data_sources.md): Origin, collection horizons, and raw measurement definitions.
- [docs/data_dictionary.md](docs/data_dictionary.md): Comprehensive schema of raw, processed, and model feature sets.
- [docs/methodology.md](docs/methodology.md): Preprocessing rules, chronological splitting, and loss formulations.
- [docs/research_decisions.md](docs/research_decisions.md): Detailed diagnostic analyses of anomalies A001 and A002.
- [docs/architecture_roadmap.md](docs/architecture_roadmap.md): Multi-target forecasting and grid optimization decision engine roadmap.
- [reports/milestones/demand_forecasting_milestone.md](reports/milestones/demand_forecasting_milestone.md): Formal research report on demand modeling results.
- [reports/milestones/renewable_forecasting_milestone.md](reports/milestones/renewable_forecasting_milestone.md): Wind and solar results with the corrected prior-day benchmark.
- [reports/audits/full_dataset_model_readiness.md](reports/audits/full_dataset_model_readiness.md): Four-year stack quality findings and requirements before retraining.
- [reports/milestones/renewable_v3_full_history.md](reports/milestones/renewable_v3_full_history.md): Four-year renewable training, corrected baseline, and selected forecasts.
