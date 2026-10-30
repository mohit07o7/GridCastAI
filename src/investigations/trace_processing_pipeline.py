
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_clean.csv"
VALIDATED_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"

START = pd.Timestamp("2024-06-06 18:00:00")
END = pd.Timestamp("2024-06-10 08:00:00")

columns = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

clean = pd.read_csv(CLEAN_PATH, parse_dates=["Timestamp"])
validated = pd.read_csv(VALIDATED_PATH, parse_dates=["Timestamp"])

clean = clean[clean["Timestamp"].between(START, END)].copy()
validated = validated[
    validated["Timestamp"].between(START, END)
].copy()

print("=" * 70)
print("GRIDCASTAI — PROCESSING PIPELINE TRACE")
print("=" * 70)

for name, frame in [("CLEAN", clean), ("VALIDATED", validated)]:
    print(f"\n{name} DATASET")
    print(f"Rows: {len(frame)}")

    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        print(
            f"{col}: unique={frame[col].nunique()}, "
            f"missing={frame[col].isna().sum()}"
        )

comparison = clean[["Timestamp"] + columns].merge(
    validated[["Timestamp"] + columns],
    on="Timestamp",
    how="outer",
    suffixes=("_clean", "_validated"),
    indicator=True,
)

print("\nMERGE STATUS")
print(comparison["_merge"].value_counts().to_string())

for col in columns:
    clean_col = f"{col}_clean"
    validated_col = f"{col}_validated"
    difference = comparison[clean_col] - comparison[validated_col]

    print(f"\n{col}")
    print(f"Rows differing: {int(difference.ne(0).sum())}")
    print(f"Maximum absolute difference: {difference.abs().max()}")

print("\nFIRST 10 ROWS — CLEAN VS VALIDATED")
display_columns = [
    "Timestamp",
    "Wind (MW)_clean",
    "Wind (MW)_validated",
    "Total Generation (MW)_clean",
    "Total Generation (MW)_validated",
]

print(
    comparison[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nNo files were modified.")
