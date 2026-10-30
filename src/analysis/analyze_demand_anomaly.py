
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
df = pd.read_csv(path, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

target = pd.Timestamp("2025-04-04 16:00:00")
idx = df.index[df["Timestamp"] == target][0]

measurements = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW) corrected",
    "Total Generation (MW)"
]

print("TARGET VS NEIGHBORING HOURS")
print(
    df.loc[idx - 2:idx + 2, ["Timestamp"] + measurements]
    .to_string(index=False)
)

print("\nTARGET VS MEAN OF ADJACENT HOURS")
for col in measurements:
    neighbors = df.loc[[idx - 1, idx + 1], col]
    neighbor_mean = neighbors.mean()
    target_value = df.loc[idx, col]

    print(f"\n{col}")
    print(f"Target: {target_value:,.2f} MW")
    print(f"Adjacent-hour mean: {neighbor_mean:,.2f} MW")
    print(f"Difference: {target_value - neighbor_mean:,.2f} MW")
    print(f"Percentage difference: {(target_value / neighbor_mean - 1) * 100:.2f}%")
