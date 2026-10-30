
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data/raw/January 2024- June 2025.xlsx"
OUTPUT_PATH = (
    ROOT / "reports/investigations/"
    "nighttime_solar_raw_source_verification.csv"
)

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {RAW_PATH}")

    df = pd.read_excel(RAW_PATH, sheet_name="Report")

    print("Raw workbook columns:", df.columns.tolist())

    required = [
        "Timestamp", "Demand (MW)", "Wind (MW)",
        "Solar (MW)", "Total Generation (MW)"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Parsed Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    print("Rows read:", len(df))
    print("Invalid timestamps:", df["Parsed Timestamp"].isna().sum())

    dates = ["2024-04-25", "2025-03-07", "2025-03-08"]
    results = []

    for date in dates:
        day = pd.Timestamp(date)
        start = day - pd.Timedelta(hours=1)
        end = day + pd.Timedelta(days=1, hours=1)

        rows = df[
            df["Parsed Timestamp"].between(start, end)
        ].copy()

        results.append(rows)

        print(f"\n--- Original Excel rows around {date} ---")
        print(rows[
            [
                "Parsed Timestamp", "Demand (MW)", "Wind (MW)",
                "Solar (MW)", "Total Generation (MW)"
            ]
        ].to_string(index=False))

    report = pd.concat(results, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(f"\nVerification report saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
