
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Full stacked dataset: Sep 2021 – Jun 2025 (all legacy + modern, hourly)
source = PROJECT_ROOT / "data" / "processed" / "full_hourly_stacked.csv"
output = PROJECT_ROOT / "data" / "processed" / "demand_model_data.csv"

df = pd.read_csv(source, parse_dates=["timestamp"])

# ── Rename to match downstream model expectations ─────────────────────────────
df = df.rename(columns={"timestamp": "Timestamp"})

# ── Coerce key columns to numeric ─────────────────────────────────────────────
for col in ["demand_mw", "wind_mw", "solar_mw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Drop rows without a valid timestamp or demand value ───────────────────────
df = (
    df.dropna(subset=["Timestamp", "demand_mw"])
      .sort_values("Timestamp")
      .drop_duplicates(subset=["Timestamp"])
      .reset_index(drop=True)
)

# ── Verify hourly continuity ──────────────────────────────────────────────────
gaps = df["Timestamp"].diff().dropna()
non_hourly = int((gaps != pd.Timedelta(hours=1)).sum())
if non_hourly > 0:
    raise ValueError(
        f"full_hourly_stacked.csv contains {non_hourly} non-hourly gaps. "
        "Re-run resample_to_hourly.py before proceeding."
    )

# ── Known anomaly flag ────────────────────────────────────────────────────────
# Preserve the full time series; flag rather than drop the known bad reading.
df["known_demand_anomaly"] = (
    df["Timestamp"] == pd.Timestamp("2025-04-04 16:00:00")
)

# ── Calendar features ─────────────────────────────────────────────────────────
df["hour"]        = df["Timestamp"].dt.hour
df["day_of_week"] = df["Timestamp"].dt.dayofweek
df["month"]       = df["Timestamp"].dt.month

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    "Timestamp",
    "demand_mw",      # target variable
    "wind_mw",        # exogenous regressor
    "solar_mw",       # exogenous regressor (clipped, physically valid)
    "hour",
    "day_of_week",
    "month",
    "known_demand_anomaly",
]

output.parent.mkdir(parents=True, exist_ok=True)
df[OUTPUT_COLS].to_csv(output, index=False)

print("Demand model dataset prepared.")
print(f"Source  : {source.relative_to(PROJECT_ROOT)}")
print(f"Output  : {output.relative_to(PROJECT_ROOT)}")
print(f"Rows    : {len(df):,}")
print(f"Start   : {df['Timestamp'].min()}")
print(f"End     : {df['Timestamp'].max()}")
print(f"Known anomaly rows : {int(df['known_demand_anomaly'].sum())}")
print(f"Non-hourly gaps    : 0 (verified)")
print(f"\nMissing values:")
print(df[["demand_mw", "wind_mw", "solar_mw"]].isna().sum().to_string())
print("\nSource data unchanged.")
