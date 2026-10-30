
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
source = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_clean.csv"
destination = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"

df = pd.read_csv(source, parse_dates=["Timestamp"])

# Preserve reported values and create a physically constrained feature
df["Solar (MW) reported"] = df["Solar (MW)"]
df["Solar (MW) corrected"] = df["Solar (MW)"].clip(lower=0)

adjusted = df["Solar (MW)"] < 0

print("Rows checked:", len(df))
print("Negative reported solar values:", int(adjusted.sum()))
print("Values changed to zero:", int(adjusted.sum()))
print(
    "Minimum corrected solar:",
    df["Solar (MW) corrected"].min()
)
print(
    "Negative values remaining in corrected column:",
    int((df["Solar (MW) corrected"] < 0).sum())
)

# Save a separate dataset; retain the original reported solar column
df.to_csv(destination, index=False)

print("\nSaved validated dataset to:", destination)
