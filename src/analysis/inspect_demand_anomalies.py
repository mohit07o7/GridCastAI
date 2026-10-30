
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "raw" / "January 2024- June 2025.xlsx"

df = pd.read_excel(path, sheet_name="Report")
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    dayfirst=True,
    errors="coerce"
)

target = pd.Timestamp("2025-04-04 16:00:00")

# Inspect the original record and neighboring hours
mask = df["Timestamp"].between(
    target - pd.Timedelta(hours=3),
    target + pd.Timedelta(hours=3)
)

print("ORIGINAL EXCEL RECORDS AROUND THE ANOMALY")
print(
    df.loc[mask].to_string(index=False)
)

print("\nTARGET RECORD")
print(
    df.loc[df["Timestamp"] == target].to_string(index=False)
)
