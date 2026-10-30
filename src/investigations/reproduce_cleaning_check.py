
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_clean.csv"

START = pd.Timestamp("2024-06-06 18:00:00")
END = pd.Timestamp("2024-06-10 08:00:00")

measurement_columns = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

# Read the original workbook.
raw = pd.read_excel(RAW_PATH, sheet_name="Report")

# Reproduce the current clean_hourly_data.py logic exactly.
reproduced = raw.copy()
reproduced["Timestamp"] = pd.to_datetime(
    reproduced["Timestamp"],
    dayfirst=True,
    errors="coerce",
)
reproduced = reproduced.dropna(subset=["Timestamp"]).copy()

for column in measurement_columns:
    reproduced[column] = pd.to_numeric(
        reproduced[column], errors="coerce"
    )

reproduced = reproduced.sort_values("Timestamp").reset_index(drop=True)

# Read the existing clean CSV.
existing = pd.read_csv(CLEAN_PATH, parse_dates=["Timestamp"])

def select_window(frame):
    return frame.loc[
        frame["Timestamp"].between(START, END),
        ["Timestamp"] + measurement_columns,
    ].copy()

reproduced_window = select_window(reproduced)
existing_window = select_window(existing)

comparison = reproduced_window.merge(
    existing_window,
    on="Timestamp",
    how="outer",
    suffixes=("_reproduced", "_existing"),
    indicator=True,
)

print("=" * 70)
print("GRIDCASTAI — CLEANING REPRODUCTION CHECK")
print("=" * 70)

print(f"Reproduced rows in window: {len(reproduced_window)}")
print(f"Existing clean rows in window: {len(existing_window)}")
print("\nTimestamp merge status:")
print(comparison["_merge"].value_counts().to_string())

for column in measurement_columns:
    left = comparison[f"{column}_reproduced"]
    right = comparison[f"{column}_existing"]
    difference = left - right

    print(f"\n{column}")
    print(f"Reproduced unique values: {left.nunique()}")
    print(f"Existing clean unique values: {right.nunique()}")
    print(f"Rows with different values: {int(difference.ne(0).sum())}")
    print(f"Maximum absolute difference: {difference.abs().max()}")

print("\nFIRST 10 ROWS — REPRODUCED VALUES")
print(
    reproduced_window[
        ["Timestamp", "Wind (MW)", "Total Generation (MW)"]
    ].head(10).to_string(index=False)
)

print("\nFIRST 10 ROWS — EXISTING CLEAN VALUES")
print(
    existing_window[
        ["Timestamp", "Wind (MW)", "Total Generation (MW)"]
    ].head(10).to_string(index=False)
)

print("\nNo files were modified.")
