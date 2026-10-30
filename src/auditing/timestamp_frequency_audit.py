"""
timestamp_frequency_audit.py
─────────────────────────────────────────────────────────────────────────────
Deep timestamp-frequency audit across every raw workbook.

What it checks per file
───────────────────────
  • Dominant sampling interval (mode of consecutive gaps)
  • Human-readable label  (10 s / 5 min / 1 h)
  • All distinct gap sizes and their row counts
  • Expected tick count for the calendar month at that interval
  • Actual tick count (unique timestamps)
  • Missing ticks  = expected − actual  (coverage gap)
  • Duplicate timestamps
  • Irregular gaps  (gaps that differ from the dominant interval)
  • First / last timestamp

Outputs
───────
  reports/audits/timestamp_frequency_audit.csv   – one row per file
  reports/audits/timestamp_gap_detail.csv         – all distinct gaps per file
  Console summary table
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
REPORT_DIR   = PROJECT_ROOT / "reports" / "audits"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT    = REPORT_DIR / "timestamp_frequency_audit.csv"
GAP_DETAIL_OUT = REPORT_DIR / "timestamp_gap_detail.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────
INTERVAL_LABELS = {
    10:    "10 s",
    30:    "30 s",
    60:    "1 min",
    300:   "5 min",
    600:   "10 min",
    900:   "15 min",
    1800:  "30 min",
    3600:  "1 h",
}


def label_interval(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    s = int(round(seconds))
    if s in INTERVAL_LABELS:
        return INTERVAL_LABELS[s]
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h"


def expected_ticks(start: datetime, end: datetime, interval_s: float) -> int:
    """Number of ticks a perfectly regular series would have."""
    if interval_s <= 0:
        return 0
    span = (end - start).total_seconds()
    return math.floor(span / interval_s) + 1


def parse_timestamps(file: Path) -> list[datetime]:
    """
    Open the workbook read-only and extract all parseable timestamps
    from the first column that looks like a time/date field.
    Handles both SCADA two-row headers and standard single-row headers.
    """
    from openpyxl import load_workbook as opxl_load

    TIME_NAMES = {"time", "timestamp", "datetime", "date"}

    wb = opxl_load(file, read_only=True, data_only=True)
    try:
        visible = [ws for ws in wb.worksheets if ws.sheet_state == "visible"]
        if not visible:
            return []

        ws = visible[0]
        rows = ws.iter_rows(values_only=True)

        header = next(rows, None)
        if not header:
            return []

        # Detect time column index from the first row
        time_idx = next(
            (i for i, h in enumerate(header)
             if str(h).strip().lower() in TIME_NAMES),
            None,
        )

        if time_idx is None:
            # Possibly a two-row SCADA header — peek at row 2
            label_row = next(rows, None)
            if label_row:
                time_idx = next(
                    (i for i, h in enumerate(label_row)
                     if str(h).strip().lower() in TIME_NAMES),
                    None,
                )
            if time_idx is None:
                return []
        else:
            # Standard header — row 2 is already data; skip the label-scan
            pass

        timestamps: list[datetime] = []
        for row in rows:
            if not row:
                continue
            value = row[time_idx] if time_idx < len(row) else None
            if isinstance(value, datetime):
                timestamps.append(value)
            elif value is not None:
                parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
                if not pd.isna(parsed):
                    timestamps.append(parsed.to_pydatetime())

        return timestamps

    finally:
        wb.close()


# ── Main audit loop ───────────────────────────────────────────────────────────
files = sorted(
    p for p in RAW_DIR.glob("*")
    if p.suffix.lower() in {".xlsx", ".xls"}
    and not p.name.startswith("~$")
)

if not files:
    sys.exit(f"No Excel workbooks found in {RAW_DIR.resolve()}")

print(f"Timestamp-frequency audit -- {len(files)} workbooks\n{'─'*70}")

summary_rows: list[dict] = []
gap_detail_rows: list[dict] = []

for idx, file in enumerate(files, start=1):
    print(f"[{idx:2d}/{len(files)}] {file.name}", end=" ... ", flush=True)

    try:
        timestamps = parse_timestamps(file)
        timestamps.sort()

        if len(timestamps) < 2:
            print("WARN  too few timestamps")
            summary_rows.append({
                "file": file.name,
                "status": "WARN",
                "notes": "fewer than 2 parseable timestamps",
                "dominant_interval_s": None,
                "interval_label": "unknown",
                "first_timestamp": timestamps[0] if timestamps else None,
                "last_timestamp": timestamps[-1] if timestamps else None,
                "unique_timestamps": len(set(timestamps)),
                "duplicate_timestamps": len(timestamps) - len(set(timestamps)),
                "expected_ticks": None,
                "missing_ticks": None,
                "coverage_pct": None,
                "irregular_gaps": None,
                "distinct_gap_count": 0,
            })
            continue

        # ── Gap analysis ──────────────────────────────────────────────────
        gaps_s = [
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        ]
        gap_counter = Counter(gaps_s)

        dominant_s, dominant_count = gap_counter.most_common(1)[0]
        interval_lbl = label_interval(dominant_s)

        # All distinct gaps for the detail table
        for gap_val, gap_count in sorted(gap_counter.items()):
            gap_detail_rows.append({
                "file": file.name,
                "gap_seconds": gap_val,
                "gap_label": label_interval(gap_val),
                "occurrences": gap_count,
                "pct_of_gaps": round(100 * gap_count / len(gaps_s), 3),
                "is_dominant": gap_val == dominant_s,
            })

        # ── Completeness ──────────────────────────────────────────────────
        unique_ts  = set(timestamps)
        n_unique   = len(unique_ts)
        n_dupes    = len(timestamps) - n_unique
        n_expected = expected_ticks(timestamps[0], timestamps[-1], dominant_s)
        n_missing  = max(0, n_expected - n_unique)
        coverage   = round(100 * n_unique / n_expected, 3) if n_expected else None

        # Irregular gaps = gaps that differ from the dominant interval
        # Allow +-0.5 s tolerance for sub-second clock drift
        n_irregular = sum(
            cnt for gap, cnt in gap_counter.items()
            if abs(gap - dominant_s) > 0.5
        )

        # ── Status ────────────────────────────────────────────────────────
        issues = []
        if n_dupes > 0:
            issues.append(f"{n_dupes:,} duplicate timestamps")
        if n_missing > 0:
            issues.append(f"{n_missing:,} missing ticks")
        if n_irregular > 0:
            issues.append(f"{n_irregular:,} irregular gaps")

        status = "WARN" if issues else "OK"
        notes  = "; ".join(issues)

        summary_rows.append({
            "file": file.name,
            "status": status,
            "notes": notes,
            "dominant_interval_s": dominant_s,
            "interval_label": interval_lbl,
            "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
            "unique_timestamps": n_unique,
            "duplicate_timestamps": n_dupes,
            "expected_ticks": n_expected,
            "missing_ticks": n_missing,
            "coverage_pct": coverage,
            "irregular_gaps": n_irregular,
            "distinct_gap_count": len(gap_counter),
        })

        print(
            f"OK  [{interval_lbl}]  "
            f"{n_unique:,} unique  "
            f"missing={n_missing:,}  "
            f"irregular={n_irregular:,}  "
            f"dupes={n_dupes:,}"
        )

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        summary_rows.append({
            "file": file.name,
            "status": "ERROR",
            "notes": str(exc),
            "dominant_interval_s": None,
            "interval_label": "unknown",
            "first_timestamp": None,
            "last_timestamp": None,
            "unique_timestamps": None,
            "duplicate_timestamps": None,
            "expected_ticks": None,
            "missing_ticks": None,
            "coverage_pct": None,
            "irregular_gaps": None,
            "distinct_gap_count": None,
        })

# ── Save reports ──────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)

gap_df = pd.DataFrame(gap_detail_rows)
gap_df.to_csv(GAP_DETAIL_OUT, index=False)

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TIMESTAMP FREQUENCY AUDIT -- SUMMARY")
print(f"{'='*70}\n")

# Group by interval label
freq_groups = (
    summary_df.groupby("interval_label", dropna=False)["file"]
    .apply(list)
    .to_dict()
)

for lbl, group_files in sorted(freq_groups.items()):
    print(f"  [{lbl}]  ({len(group_files)} file{'s' if len(group_files) != 1 else ''})")
    for fn in sorted(group_files):
        row = summary_df.loc[summary_df["file"] == fn].iloc[0]
        mark = "WARN " if row["status"] != "OK" else "     "
        coverage_val = row['coverage_pct']
        coverage_str = f"{coverage_val:>7.3f}%" if coverage_val is not None else "    n/a"
        print(
            f"      {mark}{fn:<35s}"
            f"  coverage={coverage_str}"
            f"  missing={str(int(row['missing_ticks']) if row['missing_ticks'] is not None else 0):>6}"
            f"  irregular={str(int(row['irregular_gaps']) if row['irregular_gaps'] is not None else 0):>6}"
        )
    print()

print(f"\nStatus breakdown:")
print(summary_df["status"].value_counts(dropna=False).to_string())

print(f"\n{'─'*70}")
print(f"Summary    -> {SUMMARY_OUT}")
print(f"Gap detail -> {GAP_DETAIL_OUT}")
print("Raw workbooks were NOT modified.")
