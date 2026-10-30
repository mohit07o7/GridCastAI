"""Verify selected hourly quality flags against original legacy workbooks.

Example:
    python src/investigations/verify_legacy_source_anomalies.py \
        --raw-dir /path/to/GridCastAI/data/raw
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "reports" / "audits"


def source_rows(path: Path, last_hour: datetime):
    sheet = load_workbook(path, read_only=True, data_only=True)["Sheet1"]
    for row in sheet.iter_rows(min_row=3, max_col=11, values_only=True):
        stamp = row[0]
        if not isinstance(stamp, datetime):
            continue
        if stamp >= last_hour + timedelta(hours=1):
            break
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    zeros = pd.read_csv(AUDITS / "full_dataset_all_zero_hours.csv", parse_dates=["timestamp"])
    details = []
    for filename, group in zeros.groupby("source_file"):
        expected = {
            stamp.to_pydatetime(): int(count)
            for stamp, count in zip(group["timestamp"], group["rows_per_hour"])
        }
        actual = defaultdict(int)
        nonzero = defaultdict(int)
        for row in source_rows(args.raw_dir / filename, max(expected)):
            hour = row[0].replace(minute=0, second=0, microsecond=0)
            if hour not in expected:
                continue
            actual[hour] += 1
            nonzero[hour] += sum(value not in (0, None) for value in row[5:11])
        for hour, expected_count in expected.items():
            details.append({
                "source_file": filename,
                "hour": str(hour),
                "expected_source_rows": expected_count,
                "observed_source_rows": actual[hour],
                "nonzero_measurement_cells_F_to_K": nonzero[hour],
                "verified_source_zero": actual[hour] == expected_count and nonzero[hour] == 0,
            })

    october = args.raw_dir / "October 2022.xlsx"
    october_results = {}
    for hour in [datetime(2022, 10, 1, 12), datetime(2022, 10, 1, 13)]:
        october_results[str(hour)] = {"source_rows": 0, "component_zero_combined_positive_rows": 0}
    first_bad = last_bad = None
    for row in source_rows(october, datetime(2022, 10, 1, 13)):
        stamp = row[0]
        hour = stamp.replace(minute=0, second=0, microsecond=0)
        key = str(hour)
        if key not in october_results:
            continue
        october_results[key]["source_rows"] += 1
        if row[5] == 0 and row[7] == 0 and isinstance(row[8], (int, float)) and row[8] > 0:
            october_results[key]["component_zero_combined_positive_rows"] += 1
            first_bad = stamp if first_bad is None else first_bad
            last_bad = stamp

    modern = load_workbook(
        args.raw_dir / "January 2024- June 2025.xlsx", read_only=True, data_only=True
    )["Report"]
    modern_flatline = []
    for row in modern.iter_rows(min_row=2, max_col=5, values_only=True):
        try:
            stamp = datetime.strptime(row[0], "%d-%m-%Y %H:%M:%S")
        except (TypeError, ValueError):
            continue
        if datetime(2025, 3, 6, 12) <= stamp <= datetime(2025, 3, 8, 11):
            modern_flatline.append(row[3])

    result = {
        "raw_dir": str(args.raw_dir),
        "all_zero_hour_verification": details,
        "all_12_hours_verified": len(details) == 12 and all(d["verified_source_zero"] for d in details),
        "october_2022_component_dropout": october_results,
        "component_dropout_first": str(first_bad),
        "component_dropout_last": str(last_bad),
        "modern_solar_flatline": {
            "start": "2025-03-06 12:00:00",
            "end": "2025-03-08 11:00:00",
            "source_rows": len(modern_flatline),
            "unique_solar_values_mw": sorted(set(modern_flatline)),
            "verified": len(modern_flatline) == 48 and set(modern_flatline) == {64300.61},
        },
    }
    output = AUDITS / "legacy_source_anomaly_verification.json"
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
