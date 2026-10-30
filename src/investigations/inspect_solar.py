
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_clean.csv"
df = pd.read_csv(path, parse_dates=["Timestamp"])

negative = df[df["Solar (MW)"] < 0].copy()

print("Negative solar rows:", len(negative))
print("\nSolar value distribution for negative readings:")
print(negative["Solar (MW)"].describe())

print("\nFirst 20 negative solar records:")
print(
    negative[["Timestamp", "Demand (MW)", "Wind (MW)", "Solar (MW)"]]
    .head(20)
    .to_string(index=False)
)

print("\nNegative readings by hour of day:")
print(negative.groupby(negative["Timestamp"].dt.hour).size())

print("\nNegative readings by month:")
print(negative.groupby(negative["Timestamp"].dt.to_period("M")).size())
