
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp")

# Define provisional net grid requirement
df["Grid Requirement (MW)"] = (
    df["Demand (MW)"]
    - df["Wind (MW)"]
    - df["Solar (MW) corrected"]
)

# 1. Full-period demand
plt.figure(figsize=(14, 5))
plt.plot(df["Timestamp"], df["Demand (MW)"], linewidth=0.7)
plt.title("Hourly Electricity Demand — India Dataset")
plt.xlabel("Timestamp")
plt.ylabel("Demand (MW)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hourly_demand.png", dpi=150)
plt.show()

# 2. One representative week
week_start = df["Timestamp"].min()
week_end = week_start + pd.Timedelta(days=7)
week = df[
    (df["Timestamp"] >= week_start)
    & (df["Timestamp"] < week_end)
]

plt.figure(figsize=(14, 5))
plt.plot(week["Timestamp"], week["Demand (MW)"], label="Demand")
plt.plot(week["Timestamp"], week["Wind (MW)"], label="Wind")
plt.plot(
    week["Timestamp"],
    week["Solar (MW) corrected"],
    label="Solar"
)
plt.title("Demand and Renewable Generation — First Week")
plt.xlabel("Timestamp")
plt.ylabel("Power (MW)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "first_week_energy.png", dpi=150)
plt.show()

# 3. Daily average profiles
df["Hour"] = df["Timestamp"].dt.hour
hourly_profile = df.groupby("Hour")[
    ["Demand (MW)", "Wind (MW)", "Solar (MW) corrected",
     "Grid Requirement (MW)"]
].mean()

plt.figure(figsize=(12, 5))
for column in hourly_profile.columns:
    plt.plot(hourly_profile.index, hourly_profile[column], label=column)

plt.title("Average Hour-of-Day Profile")
plt.xlabel("Hour of Day")
plt.ylabel("Average Power (MW)")
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hourly_profiles.png", dpi=150)
plt.show()

print(f"EDA plots saved in: {FIGURES_DIR}")
print("\nAverage hourly profile:")
print(hourly_profile.round(2).to_string())
