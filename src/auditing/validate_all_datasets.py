
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "audits"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TIME_NAMES = {"time", "timestamp", "datetime", "date"}

def find_time_column(columns):
    for col in columns:
        if str(col).strip().lower() in TIME_NAMES:
            return col
    return None


def audit_workbook(path):
    result = {
        "file": path.name,
        "status": "OK",
        "sheet": None,
        "format": None,
        "rows": 0,
        "invalid_timestamps": None,
        "duplicate_timestamps": None,
        "missing_cells": None,
        "non_numeric_measurements": None,
        "negative_numeric_values": None,
        "large_hourly_jumps": None,
        "notes": "",
    }

    try:
        xls = pd.ExcelFile(path)
        sheet = "Report" if "Report" in xls.sheet_names else "Sheet1"
        if sheet not in xls.sheet_names:
            sheet = xls.sheet_names[0]

        result["sheet"] = sheet

        # Inspect the first two rows to detect the file format
        preview = pd.read_excel(
            path, sheet_name=sheet, header=None, nrows=2
        )

        first_row = preview.iloc[0].astype(str).str.strip().str.lower()
        second_row = (
            preview.iloc[1].astype(str).str.strip().str.lower()
            if len(preview) > 1
            else pd.Series(dtype=str)
        )

        if "timestamp" in first_row.values:
            # Standard hourly report: first row contains column names
            df = pd.read_excel(path, sheet_name=sheet)
            time_col = next(
                c for c in df.columns
                if str(c).strip().lower() == "timestamp"
            )
            result["format"] = "standard_hourly"

        elif "time" in first_row.values:
            # SCADA format: first row has signal IDs; second row has labels
            df = pd.read_excel(
                path, sheet_name=sheet, header=None, skiprows=2
            )
            df.columns = [
                f"column_{i}" for i in range(df.shape[1])
            ]
            time_col = "column_0"
            result["format"] = "scada_two_row_header"

        else:
            result["status"] = "REVIEW"
            result["notes"] = "Unrecognized header format"
            return result

        result["rows"] = len(df)

        timestamps = pd.to_datetime(
            df[time_col], errors="coerce", dayfirst=True
        )

        result["invalid_timestamps"] = int(timestamps.isna().sum())
        valid_ts = timestamps.dropna()
        result["duplicate_timestamps"] = int(
            valid_ts.duplicated().sum()
        )
        result["missing_cells"] = int(df.isna().sum().sum())

        measurement_cols = [
            c for c in df.columns if c != time_col
        ]
        numeric_data = {}
        non_numeric_count = 0
        negative_count = 0
        jump_count = 0

        for col in measurement_cols:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().any():
                numeric_data[col] = converted
                non_numeric_count += int(
                    (df[col].notna() & converted.isna()).sum()
                )
                negative_count += int((converted < 0).sum())

        result["non_numeric_measurements"] = non_numeric_count
        result["negative_numeric_values"] = negative_count

        if numeric_data:
            work = pd.DataFrame(numeric_data)
            work["_timestamp"] = timestamps
            work = work.dropna(subset=["_timestamp"])
            work = work.sort_values("_timestamp")

            for col in numeric_data:
                previous = work[col].shift(1)
                gap = work["_timestamp"].diff()
                comparable = (
                    gap.eq(pd.Timedelta(hours=1))
                    & previous.notna()
                    & work[col].notna()
                    & previous.ne(0)
                )
                pct_change = (
                    (work[col] - previous).abs()
                    / previous.abs()
                )
                jump_count += int(
                    (comparable & (pct_change > 0.50)).sum()
                )

        result["large_hourly_jumps"] = jump_count

        issues = (
            result["invalid_timestamps"]
            or result["duplicate_timestamps"]
            or result["missing_cells"]
            or result["non_numeric_measurements"]
            or result["negative_numeric_values"]
            or result["large_hourly_jumps"]
        )

        if issues:
            result["status"] = "REVIEW"

    except Exception as exc:
        result["status"] = "ERROR"
        result["notes"] = str(exc)

    return result


files = sorted(
    p for p in RAW_DIR.glob("*")
    if p.suffix.lower() in {".xlsx", ".xls"}
    and not p.name.startswith("~$")
)

if not files:
    raise FileNotFoundError(
        f"No Excel workbooks found in {RAW_DIR.resolve()}"
    )

results = []

for i, path in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] Auditing {path.name}")
    results.append(audit_workbook(path))

report = pd.DataFrame(results)
output = REPORT_DIR / "automated_quality_report.csv"
report.to_csv(output, index=False)

print("\nAUDIT SUMMARY")
print(report["status"].value_counts(dropna=False).to_string())
print("\nReport saved to:", output)
print("\nFiles needing attention:")
print(
    report.loc[
        report["status"] != "OK",
        ["file", "status", "notes"]
    ].to_string(index=False)
)
