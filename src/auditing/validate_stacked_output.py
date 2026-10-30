import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── 1. Run log: all statuses + October 2022 residual ─────────────────────────
log = pd.read_csv(ROOT / "reports/audits/resample_run_log.csv")

print("=== STATUS BREAKDOWN ===")
print(log["status"].value_counts(dropna=False).to_string())

print("\n=== SOLAR_WIND_RESIDUAL_BAD_HOURS (all files) ===")
print(log[["file", "solar_wind_residual_bad_hours"]].to_string(index=False))

oct22 = log[log["file"] == "October 2022.xlsx"].iloc[0]
print("\n=== OCTOBER 2022 DETAIL ===")
for col in log.columns:
    print(f"  {col:<35s}: {oct22[col]}")

# ── 2. full_hourly_stacked.csv: duplicates + gap check ───────────────────────
print("\n\n=== FULL STACKED DATASET VALIDATION ===")
df = pd.read_csv(
    ROOT / "data/processed/full_hourly_stacked.csv",
    parse_dates=["timestamp"],
)

print(f"Total rows          : {len(df):,}")
print(f"Date range          : {df['timestamp'].min()} -> {df['timestamp'].max()}")

dupes = int(df["timestamp"].duplicated().sum())
print(f"Duplicate timestamps: {dupes}")

df = df.sort_values("timestamp").reset_index(drop=True)
gaps = df["timestamp"].diff().dt.total_seconds().dropna()
non_hourly = gaps[gaps != 3600]
print(f"Non-1-hour gaps     : {len(non_hourly)}")

if len(non_hourly) > 0:
    print("\nDistinct non-hourly gap sizes and counts:")
    vc = non_hourly.value_counts().rename("count").reset_index()
    vc.columns = ["gap_seconds", "count"]
    print(vc.head(10).to_string(index=False))

    print("\nTimestamps with a non-hourly gap immediately before them:")
    gap_ts = df.loc[non_hourly.index, ["timestamp"]].copy()
    gap_ts["gap_seconds"] = non_hourly.values
    print(gap_ts.head(10).to_string(index=False))

print("\n=== MISSING VALUES IN KEY COLUMNS ===")
key_cols = ["wind_mw", "solar_mw", "solar_raw_mw", "total_gen_mw", "demand_mw"]
print(df[key_cols].isna().sum().to_string())
