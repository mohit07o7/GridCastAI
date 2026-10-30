
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"

print("Workbook path:", path.resolve())
print("Workbook exists:", path.exists())
if path.exists():
    print("Workbook size (bytes):", path.stat().st_size)
    print("Last modified:", pd.Timestamp(path.stat().st_mtime, unit="s"))

df = pd.read_excel(path, sheet_name="Report")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

start = pd.Timestamp("2024-06-06 18:00:00")
end = pd.Timestamp("2024-06-10 08:00:00")

window = df[df["Timestamp"].between(start, end)].copy()

for column in [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]:
    window[column] = pd.to_numeric(window[column], errors="coerce")
    print(f"{column}: {window[column].nunique()} unique values")

print("\nFirst 10 rows read directly from the workbook:")
print(
    window[
        ["Timestamp", "Wind (MW)", "Total Generation (MW)"]
    ].head(10).to_string(index=False)
)
