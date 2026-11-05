
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/hourly_grid_data_validated.csv")
OUTPUT_PATH = Path("reports/repeated_generation_investigation.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

start = pd.Timestamp("2024-06-06 18:00:00")
end = pd.Timestamp("2024-06-10 08:00:00")

window = df[df["Timestamp"].between(start, end)].copy()

columns = [
    "Timestamp",
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

print("=" * 70)
print("GRIDCASTAI — REPEATED GENERATION INVESTIGATION")
print("=" * 70)

if window.empty:
    raise ValueError("No observations found in the requested window.")

print(f"\nWindow start: {window['Timestamp'].min()}")
print(f"Window end:   {window['Timestamp'].max()}")
print(f"Rows: {len(window)}")

print("\nUNIQUE VALUES IN THE WINDOW")
for column in columns[1:]:
    print(
        f"{column}: "
        f"{window[column].nunique(dropna=False)} unique values"
    )

print("\nMINIMUM AND MAXIMUM")
print(
    window[columns[1:]]
    .agg(["min", "max", "std"])
    .round(2)
    .to_string()
)

print("\nFIRST 5 ROWS")
print(window[columns].head().to_string(index=False))

print("\nLAST 5 ROWS")
print(window[columns].tail().to_string(index=False))

print("\nSOURCE WORKBOOK CROSS-CHECK")
source_path = Path("data/raw/January 2024- June 2025.xlsx")

if not source_path.exists():
    print(f"Source workbook not found at: {source_path}")
else:
    raw = pd.read_excel(source_path, sheet_name="Report")
    raw["Timestamp"] = pd.to_datetime(
        raw["Timestamp"], errors="coerce"
    )
    raw = raw[raw["Timestamp"].between(start, end)].copy()

    print(f"Raw workbook rows in window: {len(raw)}")

    if raw.empty:
        print("No matching rows in the raw workbook.")
    else:
        raw_columns = [
            "Timestamp",
            "Demand (MW)",
            "Wind (MW)",
            "Solar (MW)",
            "Total Generation (MW)",
        ]

        print("\nRaw workbook values:")
        print(raw[raw_columns].head(5).to_string(index=False))
        print("\nRaw workbook unique-value counts:")
        for column in raw_columns[1:]:
            print(
                f"{column}: "
                f"{raw[column].nunique(dropna=False)} unique values"
            )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
window[columns].to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved processed-window extract to: {OUTPUT_PATH}")
print("No source observations were modified.")