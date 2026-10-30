
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
source = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"
destination = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_clean.csv"

df = pd.read_excel(source, sheet_name="Report")

# Parse timestamps and remove non-data summary rows
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    dayfirst=True,
    errors="coerce"
)
df = df.dropna(subset=["Timestamp"]).copy()

# Convert measurement columns to numeric
numeric_cols = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Validate the cleaned data
print("Clean dataset shape:", df.shape)
print("\nMissing values:")
print(df[numeric_cols].isna().sum())

print("\nDuplicate timestamps:", df["Timestamp"].duplicated().sum())

print("\nMeasurement ranges after removing summary rows:")
print(df[numeric_cols].describe().to_string())

print("\nNegative measurement counts:")
print((df[numeric_cols] < 0).sum())

# Verify hourly continuity
expected = pd.date_range(
    start=df["Timestamp"].min(),
    end=df["Timestamp"].max(),
    freq="h"
)
actual = pd.DatetimeIndex(df["Timestamp"])
print("\nMissing hourly timestamps:", len(expected.difference(actual)))

# Save only if numeric measurements are complete
if df[numeric_cols].isna().any().any():
    raise ValueError("Missing numeric measurements found; investigate before saving.")

destination.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(destination, index=False)

print("\nSaved to:", destination)
