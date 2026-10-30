
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "raw_processed_comparison.csv"

start = pd.Timestamp("2024-06-06 18:00:00")
end = pd.Timestamp("2024-06-10 08:00:00")

raw = pd.read_excel(RAW_PATH, sheet_name="Report")
raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
raw = raw[raw["Timestamp"].between(start, end)].copy()

processed = pd.read_csv(PROCESSED_PATH, parse_dates=["Timestamp"])
processed = processed[
    processed["Timestamp"].between(start, end)
].copy()

columns = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

for frame in [raw, processed]:
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

comparison = raw[["Timestamp"] + columns].merge(
    processed[["Timestamp"] + columns],
    on="Timestamp",
    how="outer",
    suffixes=("_raw", "_processed"),
    indicator=True,
)

print("=" * 70)
print("GRIDCASTAI — RAW VS PROCESSED COMPARISON")
print("=" * 70)

print("\nRows by merge status:")
print(comparison["_merge"].value_counts().to_string())

for col in columns:
    raw_col = f"{col}_raw"
    processed_col = f"{col}_processed"

    difference = comparison[raw_col] - comparison[processed_col]

    print(f"\n{col}")
    print(f"Raw unique values: {comparison[raw_col].nunique()}")
    print(f"Processed unique values: {comparison[processed_col].nunique()}")
    print(f"Rows with different values: {difference.ne(0).sum()}")
    print(f"Maximum absolute difference: {difference.abs().max()}")

print("\nFIRST 10 ROWS")
display_columns = [
    "Timestamp",
    "Wind (MW)_raw",
    "Wind (MW)_processed",
    "Total Generation (MW)_raw",
    "Total Generation (MW)_processed",
]
print(comparison[display_columns].head(10).to_string(index=False))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved comparison: {OUTPUT_PATH}")
print("Neither source file was modified.")
