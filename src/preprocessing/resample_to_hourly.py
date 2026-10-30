"""
resample_to_hourly.py
─────────────────────────────────────────────────────────────────────────────
Aggregate all raw legacy SCADA workbooks (5-min and 10-second) to hourly
resolution and stack them with the already-hourly Jan 2024–Jun 2025 file.

Era map (confirmed by timestamp_frequency_audit.csv):
    5 min  → Sep 2021 – Sep 2022  (12 rows/hour)
    10 s   → Oct 2022 – Dec 2023  (360 rows/hour)
    1 h    → Jan 2024 – Jun 2025  (pass-through)

Column issues handled:
    • "Wind+Solar"  (Feb 2023, May 2023)  → normalised to "Solar+Wind"
    • Negative raw solar  → clipped to 0 before averaging
    • Jan 2024–Jun 2025   → different sheet/schema; merged separately

Outputs
───────
  data/interim/legacy_monthly/   – one CSV per legacy workbook (hourly)
  data/processed/legacy_hourly_stacked.csv  – all 28 legacy months stacked
  data/processed/full_hourly_stacked.csv    – legacy + 2024-25, full span
  reports/audits/resample_run_log.csv       – per-file diagnostics
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim" / "legacy_monthly"
PROC_DIR    = ROOT / "data" / "processed"
REPORT_DIR  = ROOT / "reports" / "audits"

for d in (INTERIM_DIR, PROC_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Known interval map (seconds) ──────────────────────────────────────────────
# Derived from confirmed audit: Oct 2022 is the 10-s switchover point.
def expected_interval(filename: str) -> int:
    """Return the known source interval in seconds for a legacy SCADA file."""
    name = filename.lower()
    # Months that are confirmed 5-min (300 s)
    five_min_months = {
        "september 2021", "october 2021", "november 2021", "december 2021",
        "january 2022",   "february 2022", "march 2022",   "april 2022",
        "may 2022",       "june 2022",     "july 2022",    "august 2022",
        "september 2022",
    }
    for m in five_min_months:
        if name.startswith(m):
            return 300
    return 10  # all others are 10-second


# ── Column aliases ─────────────────────────────────────────────────────────────
# "Wind+Solar" appeared in Feb 2023 and May 2023 instead of "Solar+Wind"
SOLAR_WIND_ALIASES = {"Solar+Wind", "Wind+Solar"}

# Canonical output column names
OUT_COLS = [
    "timestamp",
    "wind_mw",
    "solar_raw_mw",       # ALL_IND_SOLAR|P  (may be negative; not clipped)
    "solar_mw",           # derived Solar col, clipped to 0
    "solar_wind_mw",      # Solar+Wind combined column from source
    "total_gen_mw",
    "demand_mw",
    "source_interval_s",
    "rows_per_hour",
    "expected_rows_per_hour",
    "source_file",
]


# ── Legacy SCADA processor ────────────────────────────────────────────────────
def process_legacy_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read one legacy SCADA workbook, resample to hourly, return
    (hourly_df, log_dict).
    """
    log = {
        "file": path.name,
        "status": "OK",
        "source_interval_s": None,
        "source_rows": None,
        "hourly_rows": None,
        "incomplete_bins": None,
        "solar_wind_alias_used": None,
        "negative_raw_solar_rows": None,
        "solar_clipped_rows": None,
        "missing_hourly_values": None,
        "solar_wind_residual_bad_hours": None,
        "notes": "",
    }

    interval_s = expected_interval(path.name)
    log["source_interval_s"] = interval_s

    # ── Read raw data ─────────────────────────────────────────────────────────
    df = pd.read_excel(
        path,
        sheet_name="Sheet1",
        header=1,          # row 0 = signal IDs, row 1 = labels (our header)
        usecols="A,F:K",
        engine="openpyxl",
    )

    # Column A is the timestamp; rename it
    df = df.rename(columns={df.columns[0]: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    log["source_rows"] = len(df)

    # ── Resolve Solar+Wind alias ──────────────────────────────────────────────
    sw_col = next((c for c in df.columns if c in SOLAR_WIND_ALIASES), None)
    if sw_col is None:
        raise ValueError(
            f"No Solar+Wind column found. Columns: {df.columns.tolist()}"
        )
    log["solar_wind_alias_used"] = sw_col

    # ── Validate expected columns ─────────────────────────────────────────────
    required = {
        "ALL_INDIA_WIND|P", "ALL_IND_SOLAR|P", "Solar", sw_col, "Total", "NLDC_DEMAND|P"
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # ── Coerce to numeric ─────────────────────────────────────────────────────
    measure_map = {
        "ALL_INDIA_WIND|P": "wind_mw",
        "ALL_IND_SOLAR|P":  "solar_raw_mw",
        "Solar":            "solar_mw",
        sw_col:             "solar_wind_mw",
        "Total":            "total_gen_mw",
        "NLDC_DEMAND|P":    "demand_mw",
    }
    df = df.rename(columns=measure_map)

    for col in measure_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Solar clipping ────────────────────────────────────────────────────────
    # solar_raw_mw: preserve original reported values (negatives kept intact).
    # solar_mw:     derived/clipped column — clip to 0 before averaging so the
    #               hourly mean reflects physical reality.
    neg_raw = int((df["solar_raw_mw"] < 0).sum())
    log["negative_raw_solar_rows"] = neg_raw

    # Do NOT clip solar_raw_mw — it must retain the original measurements.
    clipped = int((df["solar_mw"] < 0).sum())
    log["solar_clipped_rows"] = clipped
    df["solar_mw"] = df["solar_mw"].clip(lower=0)
    df["solar_wind_mw"] = df["solar_wind_mw"].clip(lower=0)

    # ── Resample to hourly ────────────────────────────────────────────────────
    indexed = df.set_index("Timestamp")
    value_cols = ["wind_mw", "solar_raw_mw", "solar_mw",
                  "solar_wind_mw", "total_gen_mw", "demand_mw"]

    hourly = indexed[value_cols].resample("1h", closed="left", label="left").mean()

    expected_per_hour = int(3600 / interval_s)
    hourly["rows_per_hour"] = (
        indexed["wind_mw"]
        .resample("1h", closed="left", label="left")
        .size()
    )
    hourly["expected_rows_per_hour"] = expected_per_hour
    hourly["source_interval_s"] = interval_s
    hourly["source_file"] = path.name

    log["hourly_rows"] = len(hourly)
    incomplete = int((hourly["rows_per_hour"] != expected_per_hour).sum())
    log["incomplete_bins"] = incomplete
    log["missing_hourly_values"] = int(hourly[value_cols].isna().sum().sum())

    # Consistency check: solar + wind should match the source combined column
    residual = (
        hourly["solar_wind_mw"]
        - (hourly["solar_mw"] + hourly["wind_mw"])
    ).abs()
    log["solar_wind_residual_bad_hours"] = int((residual > 1.0).sum())

    hourly = hourly.reset_index().rename(columns={"Timestamp": "timestamp"})

    return hourly[OUT_COLS], log


# ── Modern hourly file processor ──────────────────────────────────────────────
def process_modern_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read the Jan 2024–Jun 2025 hourly file (sheet='Report', different schema).
    Returns a DataFrame aligned to OUT_COLS with NaN for columns not present.
    """
    log = {
        "file": path.name,
        "status": "OK",
        "source_interval_s": 3600,
        "source_rows": None,
        "hourly_rows": None,
        "incomplete_bins": 0,
        "solar_wind_alias_used": "n/a (modern file)",
        "negative_raw_solar_rows": None,
        "solar_clipped_rows": None,
        "missing_hourly_values": None,
        "solar_wind_residual_bad_hours": None,
        "notes": "Modern hourly file – no resampling needed",
    }

    df = pd.read_excel(path, sheet_name="Report", engine="openpyxl")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    log["source_rows"] = len(df)

    for col in ["Demand (MW)", "Wind (MW)", "Solar (MW)", "Total Generation (MW)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_solar       = df["Solar (MW)"]
    corrected_solar = raw_solar.clip(lower=0)
    wind            = df["Wind (MW)"]

    # Log counts from the original raw series before any clipping.
    log["negative_raw_solar_rows"] = int((raw_solar < 0).sum())
    log["solar_clipped_rows"]      = int((raw_solar < 0).sum())

    out = pd.DataFrame({
        "timestamp":              df["Timestamp"],
        "wind_mw":                wind,
        "solar_raw_mw":           raw_solar,
        "solar_mw":               corrected_solar,
        "solar_wind_mw":          corrected_solar + wind,
        "total_gen_mw":           df["Total Generation (MW)"],
        "demand_mw":              df["Demand (MW)"],
        "source_interval_s":      3600,
        "rows_per_hour":          1,
        "expected_rows_per_hour": 1,
        "source_file":            path.name,
    })

    log["hourly_rows"] = len(out)
    log["missing_hourly_values"] = int(out[
        ["wind_mw","solar_mw","total_gen_mw","demand_mw"]
    ].isna().sum().sum())
    log["solar_wind_residual_bad_hours"] = 0

    return out[OUT_COLS], log


# ── Main ──────────────────────────────────────────────────────────────────────
MODERN_FILE = "January 2024- June 2025.xlsx"

legacy_files = sorted(
    p for p in RAW_DIR.glob("*.xlsx")
    if not p.name.startswith("~$") and p.name != MODERN_FILE
)
modern_file = RAW_DIR / MODERN_FILE

if not legacy_files:
    sys.exit(f"No legacy SCADA workbooks found in {RAW_DIR}")

print(f"Processing {len(legacy_files)} legacy workbooks + 1 modern file")
print("=" * 72)

all_logs: list[dict] = []
legacy_frames: list[pd.DataFrame] = []

for i, path in enumerate(legacy_files, start=1):
    print(f"[{i:2d}/{len(legacy_files)}] {path.name} … ", end="", flush=True)
    try:
        hourly_df, log = process_legacy_file(path)
        legacy_frames.append(hourly_df)

        # Save individual monthly interim file
        stem = path.stem.replace(" ", "_").lower()
        hourly_df.to_csv(INTERIM_DIR / f"{stem}_hourly.csv", index=False)

        print(
            f"OK  [{log['source_interval_s']}s → 1h]  "
            f"{log['source_rows']:,} rows → {log['hourly_rows']:,} hourly  "
            f"incomplete={log['incomplete_bins']}  "
            f"neg_solar={log['negative_raw_solar_rows']:,}"
        )
    except Exception as exc:
        log = {"file": path.name, "status": "ERROR", "notes": str(exc)}
        for k in ["source_interval_s","source_rows","hourly_rows","incomplete_bins",
                  "solar_wind_alias_used","negative_raw_solar_rows","solar_clipped_rows",
                  "missing_hourly_values","solar_wind_residual_bad_hours"]:
            log.setdefault(k, None)
        print(f"ERROR: {exc}")

    all_logs.append(log)

# ── Modern file ───────────────────────────────────────────────────────────────
print(f"\n[{'M':2s}]  {MODERN_FILE} … ", end="", flush=True)
try:
    modern_df, mod_log = process_modern_file(modern_file)
    print(f"OK  [already 1h]  {mod_log['hourly_rows']:,} hourly rows")
except Exception as exc:
    mod_log = {"file": MODERN_FILE, "status": "ERROR", "notes": str(exc)}
    modern_df = pd.DataFrame(columns=OUT_COLS)
    print(f"ERROR: {exc}")

all_logs.append(mod_log)

# ── Stack and save ────────────────────────────────────────────────────────────
legacy_stack = (
    pd.concat(legacy_frames, ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
)

LEGACY_OUT = PROC_DIR / "legacy_hourly_stacked.csv"
legacy_stack.to_csv(LEGACY_OUT, index=False)

full_stack = (
    pd.concat([legacy_stack, modern_df], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
)

FULL_OUT = PROC_DIR / "full_hourly_stacked.csv"
full_stack.to_csv(FULL_OUT, index=False)

# ── Run log ───────────────────────────────────────────────────────────────────
log_df = pd.DataFrame(all_logs)
LOG_OUT = REPORT_DIR / "resample_run_log.csv"
log_df.to_csv(LOG_OUT, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("RESAMPLE SUMMARY")
print(f"{'='*72}")
print(f"  Legacy files processed : {len(legacy_frames)}/{len(legacy_files)}")
print(f"  Legacy hourly rows     : {len(legacy_stack):,}")
print(f"  Full span hourly rows  : {len(full_stack):,}")
print(f"  Date range             : {full_stack['timestamp'].min()} → {full_stack['timestamp'].max()}")
print(f"\n  Status breakdown:")
print(log_df["status"].value_counts(dropna=False).to_string())

errors = log_df[log_df["status"] == "ERROR"]
if not errors.empty:
    print(f"\n  Files with errors:")
    for _, row in errors.iterrows():
        print(f"    {row['file']}: {row['notes']}")

print(f"\n  Outputs:")
print(f"    Monthly interim  → {INTERIM_DIR.relative_to(ROOT)}/")
print(f"    Legacy stacked   → {LEGACY_OUT.relative_to(ROOT)}")
print(f"    Full stacked     → {FULL_OUT.relative_to(ROOT)}")
print(f"    Run log          → {LOG_OUT.relative_to(ROOT)}")
print("\nRaw workbooks were NOT modified.")
