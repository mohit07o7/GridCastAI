from pathlib import Path
from collections import Counter
from openpyxl import load_workbook
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
files = sorted(RAW_DIR.rglob("*.xlsx"))
results = []

print(f"Auditing {len(files)} Excel workbooks...\n")

for file in files:
    wb = None

    try:
        wb = load_workbook(
            file,
            read_only=True,
            data_only=True
        )

        visible = [
            ws for ws in wb.worksheets
            if ws.sheet_state == "visible"
        ]

        if not visible:
            print(f"SKIP {file.name}: no visible worksheet")
            continue

        ws = visible[0]
        rows = ws.iter_rows(values_only=True)

        header = next(rows, None)
        labels = next(rows, None)

        if not header:
            print(f"SKIP {file.name}: empty worksheet")
            continue

        # Some SCADA files have field names in row 1
        # and readable signal labels in row 2.
        headers = []
        for i, value in enumerate(header):
            label = labels[i] if labels and i < len(labels) else None
            headers.append(str(label or value or f"column_{i}"))

        time_idx = next(
            (
                i for i, h in enumerate(header)
                if str(h).strip().lower() in
                {"time", "timestamp", "datetime", "date"}
            ),
            None
        )

        if time_idx is None:
            print(f"SKIP {file.name}: timestamp column not identified")
            continue

        timestamps = []
        missing = Counter()
        total = 0

        for row in rows:
            if not row:
                continue

            total += 1

            for i, name in enumerate(headers):
                value = row[i] if i < len(row) else None
                if value is None:
                    missing[name] += 1

            value = row[time_idx] if time_idx < len(row) else None

            if isinstance(value, datetime):
                timestamps.append(value)
            elif value is not None:
                parsed = pd.to_datetime(
                    value,
                    errors="coerce",
                    dayfirst=True
                )
                if not pd.isna(parsed):
                    timestamps.append(parsed.to_pydatetime())

        timestamps.sort()
        gaps = Counter(
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        )

        result = {
            "file": file.name,
            "rows": total,
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
            "unique_timestamps": len(set(timestamps)),
            "duplicate_timestamps": len(timestamps) - len(set(timestamps)),
            "most_common_interval_seconds": gaps.most_common(1)[0][0] if gaps else None,
            "missing_cells": sum(missing.values()),
            "columns": " | ".join(headers),
        }

        results.append(result)

        print(f"\nFILE: {file.name}")
        print(f"Rows: {total:,}")
        print(f"Date range: {result['first_timestamp']} -> {result['last_timestamp']}")
        print(f"Unique timestamps: {result['unique_timestamps']:,}")
        print(f"Duplicate timestamps: {result['duplicate_timestamps']:,}")
        print(f"Most common interval (seconds): {result['most_common_interval_seconds']}")
        print(f"Missing cells: {result['missing_cells']:,}")
        print("Columns:", " | ".join(headers))

    except Exception as exc:
        print(f"\nERROR in {file.name}: {type(exc).__name__}: {exc}")

    finally:
        if wb is not None:
            wb.close()

if results:
    output = PROJECT_ROOT / "reports" / "audits"
    output.mkdir(parents=True, exist_ok=True)

    summary_path = output / "data_audit_summary.csv"
    pd.DataFrame(results).to_csv(
        summary_path,
        index=False
    )

    print(f"\nSummary saved to {summary_path}")

print("\nFull audit complete. Raw workbooks were not modified.")
