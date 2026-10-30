from pathlib import Path
from openpyxl import load_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
files = sorted(RAW_DIR.rglob("*.xlsx"))

print("=" * 65)
print("GRIDCAST AI — RAW DATA AUDIT")
print("=" * 65)
print(f"Excel files found: {len(files)}")

if not files:
    print(f"No Excel files found in {RAW_DIR.resolve()}")
    raise SystemExit(1)

for file in files:
    print("\n" + "-" * 65)
    print(f"File: {file.name}")
    print(f"Size: {file.stat().st_size / (1024 * 1024):.2f} MB")

    wb = None
    try:
        wb = load_workbook(
            file,
            read_only=True,
            data_only=True
        )

        print("Sheets:", wb.sheetnames)

        visible_sheets = [
            sheet for sheet in wb.worksheets
            if sheet.sheet_state == "visible"
        ]

        if not visible_sheets:
            print("No visible worksheets found.")
            continue

        ws = visible_sheets[0]

        print(f"Worksheet: {ws.title}")
        print(
            f"Reported dimensions: {ws.max_row} rows x "
            f"{ws.max_column} columns"
        )

        print("First 5 rows:")
        for row in ws.iter_rows(
            min_row=1,
            max_row=min(5, ws.max_row),
            values_only=True
        ):
            print(row)

    except Exception as exc:
        print(f"Inspection error: {type(exc).__name__}: {exc}")

    finally:
        if wb is not None:
            wb.close()

print("\nAudit complete. No data has been modified.")
