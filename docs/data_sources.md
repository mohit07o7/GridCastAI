# Data Sources Documentation

**Project**: GridCastAI  
**Path**: `docs/data_sources.md`  
**Last Updated**: October 2026  

---

## 1. Overview of Raw Datasets

GridCastAI utilizes real-world national power grid operational logs representing hourly electricity consumption and renewable generation across India. The datasets were obtained from national grid dispatch monitoring records (Power System Operation Corporation Limited / Grid Controller of India).

Raw files are preserved immutably in `data/raw/`. Under no circumstances are raw workbooks modified in place.

---

## 2. Inventory of Source Files

### A. Primary Time Series (Consolidated)
- **Filename**: `data/raw/January 2024- June 2025.xlsx`
- **Worksheet**: `Report`
- **Time Horizon**: `2024-01-01 00:00:00` to `2025-06-24 23:00:00`
- **Observations**: 12,984 consecutive hourly timestamps
- **Format**: Standard hourly time series with summary rows at the sheet footer

### B. Historical Monthly Workbooks
Stored in `data/raw/` for multi-year historical auditing and seasonal expansion:
- `October 2021.xlsx` – `December 2021.xlsx`
- `January 2022.xlsx` – `December 2022.xlsx`
- `January 2023.xlsx` – `December 2023.xlsx`
*(Note: Files range between standard hourly sheets and two-row SCADA headers audited via `src/auditing/validate_all_datasets.py`)*

---

## 3. Measurement Definitions & Units

| Column Name | Units | Measurement Type | Description |
| :--- | :--- | :--- | :--- |
| `Timestamp` | Datetime (IST) | Temporal index | Recorded hourly timestamp (`DD-MM-YYYY HH:MM:SS`) |
| `Demand (MW)` | Megawatts (MW) | Grid Load | Instantaneous aggregate electricity demand on the transmission grid |
| `Wind (MW)` | Megawatts (MW) | Renewable Generation | Total dispatched wind turbine generation capacity |
| `Solar (MW)` | Megawatts (MW) | Renewable Generation | Total dispatched utility-scale solar photovoltaic generation |
| `Total Generation (MW)` | Megawatts (MW) | Total Supply | Aggregate electrical power dispatched from all sources (thermal, hydro, nuclear, RES) |

---

## 4. Ingestion & Quality Characteristics

1. **Footer Summary Rows**: Raw Excel sheets conclude with statistical aggregates (e.g. Max, Min, Average) containing non-datetime strings. The ingestion pipeline (`src/preprocessing/clean_hourly_data.py`) explicitly filters records using datetime coercion (`dropna(subset=['Timestamp'])`).
2. **Time Continuity**: The primary dataset represents an unbroken hourly frequency ($f = \text{1 hour}$) across 12,984 steps with zero missing hourly stamps.
3. **Immutability Guarantee**: All preprocessing transformations output to `data/processed/`, retaining `data/raw/` as an immutable source of truth for reproducibility.
