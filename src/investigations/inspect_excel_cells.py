
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"

wb = load_workbook(path, data_only=True, read_only=True)
ws = wb["Report"]

rows = list(ws.iter_rows(values_only=True))
wb.close()

df = pd.DataFrame(rows[1:], columns=rows[0])

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce",
)

target = pd.Timestamp("2024-06-07 00:00:00")
matches = df.index[df["Timestamp"].eq(target)].tolist()

print("=" * 70)
print("GRIDCASTAI — TIMESTAMP-ALIGNED EXCEL INSPECTION")
print("=" * 70)
print(f"Rows loaded: {len(df)}")
print(f"Invalid timestamps: {df['Timestamp'].isna().sum()}")
print(f"Matches for {target}: {len(matches)}")

if len(matches) != 1:
    raise ValueError("Expected exactly one matching timestamp.")

i = matches[0]

print("\nRows around June 7, 2024:")
print(
    df.iloc[max(0, i - 6):i + 4]
    .to_string(index=False)
)

print("\nUnique values in the 87-hour interval:")
window = df[
    df["Timestamp"].between(
        pd.Timestamp("2024-06-06 18:00:00"),
        pd.Timestamp("2024-06-10 08:00:00"),
    )
]

for col in [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]:
    print(f"{col}: {window[col].nunique()}")

print("\nInspection complete. No files were modified.")
