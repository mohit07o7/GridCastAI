
from pathlib import Path

import pandas as pd

# Resolve the project root from:
# GridCastAI/src/investigations/this_script.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January 2024- June 2025.xlsx"
)

RUNS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "audits"
    / "full_dataset_repeated_runs.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "repeated_generation_relationship.csv"
)

MEASUREMENTS = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

# Load the original workbook without modifying it.
df = pd.read_excel(DATA_PATH, sheet_name="Report")

# Parse timestamps using the source's known format.
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce",
)

# Convert measurement columns to numeric values.
for column in MEASUREMENTS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove non-observation rows with invalid timestamps in memory only.
df = (
    df.dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

# Load the previously generated repeated-run audit.
runs = pd.read_csv(RUNS_PATH)
runs["Start"] = pd.to_datetime(runs["Start"])
runs["End"] = pd.to_datetime(runs["End"])

results = []

for _, run in runs.iterrows():
    start = run["Start"]
    end = run["End"]

    # Include both the start and end timestamps.
    window = df.loc[df["Timestamp"].between(start, end)].copy()

    result = {
        "Repeated Column": run["Column"],
        "Start": start,
        "End": end,
        "Expected Run Hours": int(run["Hours"]),
        "Observed Rows": len(window),
        "Repeated Value (MW)": run["Value"],
    }

    for column in MEASUREMENTS:
        result[f"{column} Unique Values"] = (
            window[column].nunique(dropna=True)
        )
        result[f"{column} Missing Values"] = (
            int(window[column].isna().sum())
        )

    results.append(result)

result_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 78)
print("GRIDCASTAI — REPEATED GENERATION RELATIONSHIP ANALYSIS")
print("=" * 78)
print(f"Valid timestamped source rows: {len(df):,}")
print(f"Repeated-value intervals examined: {len(result_df)}")
print(f"Output: {OUTPUT_PATH}")
print()
print(result_df.to_string(index=False))
print()
print("Interpretation note:")
print("Unique-value counts describe variation within each interval.")
print("They do not establish whether a measurement is erroneous.")
print("The raw workbook was not modified.")
