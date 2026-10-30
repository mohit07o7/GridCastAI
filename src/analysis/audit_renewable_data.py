
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
AUDITS_DIR = PROJECT_ROOT / "reports" / "audits"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"

AUDITS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

required_columns = [
    "Timestamp",
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Solar (MW) corrected",
    "Total Generation (MW)",
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("=" * 70)
print("GRIDCASTAI — RENEWABLE DATA QUALITY AUDIT")
print("=" * 70)

# 1. Basic dataset integrity
print("\n1. DATASET OVERVIEW")
print(f"Rows: {len(df):,}")
print(f"Start timestamp: {df['Timestamp'].min()}")
print(f"End timestamp:   {df['Timestamp'].max()}")
print(f"Duplicate timestamps: {df['Timestamp'].duplicated().sum()}")

intervals = df["Timestamp"].diff().dropna().dt.total_seconds() / 3600
print(f"Non-hourly intervals: {(intervals != 1).sum()}")

# 2. Measurement ranges and invalid values
measurement_columns = [
    "Wind (MW)",
    "Solar (MW)",
    "Solar (MW) corrected",
    "Total Generation (MW)",
]

print("\n2. MEASUREMENT SUMMARY")
print(
    df[measurement_columns]
    .describe()
    .round(2)
    .to_string()
)

print("\nMissing values:")
print(df[measurement_columns].isna().sum().to_string())

print("\nNegative readings:")
for col in measurement_columns:
    print(f"{col}: {(df[col] < 0).sum()}")

# 3. Inspect negative reported solar readings by hour
negative_solar = df[df["Solar (MW)"] < 0].copy()
negative_solar["Hour"] = negative_solar["Timestamp"].dt.hour

print("\n3. NEGATIVE REPORTED SOLAR READINGS")
print(f"Count: {len(negative_solar)}")

if not negative_solar.empty:
    print(f"Minimum: {negative_solar['Solar (MW)'].min():.2f} MW")
    print(
        "\nCounts by hour:"
    )
    print(
        negative_solar["Hour"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    negative_solar[
        ["Timestamp", "Solar (MW)", "Solar (MW) corrected"]
    ].to_csv(
        AUDITS_DIR / "negative_solar_observations.csv",
        index=False,
    )

# 4. Find consecutive repeated readings
def repeated_runs(column):
    values = df[column]
    run_id = values.ne(values.shift()).cumsum()

    runs = (
        df.assign(
            Value=values,
            RunID=run_id,
        )
        .groupby("RunID", sort=False)
        .agg(
            Start=("Timestamp", "first"),
            End=("Timestamp", "last"),
            Hours=("Timestamp", "size"),
            Value=("Value", "first"),
        )
        .reset_index(drop=True)
    )

    # Keep only runs lasting at least 6 hourly observations.
    return runs[runs["Hours"] >= 6].sort_values(
        "Hours", ascending=False
    )


print("\n4. LONG REPEATED-VALUE RUNS")

run_reports = []

for column in ["Wind (MW)", "Total Generation (MW)"]:
    runs = repeated_runs(column)
    runs.insert(0, "Column", column)
    run_reports.append(runs)

    print(f"\n{column}: {len(runs)} runs of 6+ consecutive hours")

    if runs.empty:
        print("No runs found.")
    else:
        print(
            runs.head(10)
            .round({"Value": 2})
            .to_string(index=False)
        )

if run_reports:
    all_runs = pd.concat(run_reports, ignore_index=True)
    all_runs.to_csv(
        EXPERIMENTS_DIR / "renewable_repeated_value_runs.csv",
        index=False,
    )

# 5. Check the energy balance without assuming it must equal zero
df["Generation Minus Demand (MW)"] = (
    df["Total Generation (MW)"] - df["Demand (MW)"]
)

print("\n5. TOTAL GENERATION MINUS DEMAND")
print(
    df["Generation Minus Demand (MW)"]
    .describe()
    .round(2)
    .to_string()
)

# 6. Save an hourly diagnostic dataset without changing source data
diagnostic_columns = [
    "Timestamp",
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Solar (MW) corrected",
    "Total Generation (MW)",
    "Generation Minus Demand (MW)",
]

df[diagnostic_columns].to_csv(
    AUDITS_DIR / "renewable_data_diagnostic.csv",
    index=False,
)

print("\n6. REPORTS SAVED")
print(f"- {AUDITS_DIR / 'renewable_data_diagnostic.csv'}")
print(f"- {EXPERIMENTS_DIR / 'renewable_repeated_value_runs.csv'}")
if not negative_solar.empty:
    print(f"- {AUDITS_DIR / 'negative_solar_observations.csv'}")

print("\nAudit complete. The source dataset was not modified.")
