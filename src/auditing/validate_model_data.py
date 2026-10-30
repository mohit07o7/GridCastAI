
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("GRIDCASTAI — MODELLING DATA VALIDATION")
print("=" * 60)

print("\n1. DATASET SHAPE")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\n2. AVAILABLE COLUMNS")
for column in df.columns:
    print(f"- {column}")

if "Timestamp" not in df.columns:
    raise ValueError("Required column 'Timestamp' is missing.")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce", format="mixed"
)

print("\n3. TIMESTAMP QUALITY")
print(f"Invalid timestamps: {df['Timestamp'].isna().sum()}")

valid = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

print(f"Duplicate timestamps: {valid['Timestamp'].duplicated().sum()}")

if not valid.empty:
    print(f"Start: {valid['Timestamp'].min()}")
    print(f"End:   {valid['Timestamp'].max()}")

    gaps = valid["Timestamp"].drop_duplicates().diff()
    print("\nObserved timestamp intervals:")
    print(gaps.value_counts().head(10).to_string())

print("\n4. MEASUREMENT QUALITY")

measurement_candidates = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Solar (MW) reported",
    "Solar (MW) corrected",
    "Total Generation (MW)",
]

for column in measurement_candidates:
    if column not in valid.columns:
        continue

    values = pd.to_numeric(valid[column], errors="coerce")

    print(f"\n{column}")
    print(f"  Missing/nonnumeric: {values.isna().sum()}")
    print(f"  Minimum: {values.min()}")
    print(f"  Maximum: {values.max()}")
    print(f"  Negative values: {(values < 0).sum()}")

print("\n5. VALIDATION STATUS")
print("This script reports data quality; it does not modify or correct data.")
print("=" * 60)
