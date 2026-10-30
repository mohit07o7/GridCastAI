
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEST_FILES = {
    "June 2022.xlsx": 300,
    "October 2022.xlsx": 10,
}

# Excel columns:
# A = Time, F = Wind, G = raw solar, H = derived solar,
# I = Solar+Wind, J = Total, K = Demand.
USECOLS = "A,F:K"

COLUMN_MAP = {
    "ALL_INDIA_WIND|P": "Wind (MW) reported legacy",
    "ALL_IND_SOLAR|P": "Solar (MW) reported legacy",
    "Solar": "Solar (MW) derived legacy",
    "Solar+Wind": "Solar+Wind (MW) source-derived",
    "Total": "Total Generation (MW) source-derived",
    "NLDC_DEMAND|P": "Demand (MW) legacy",
}


def main():
    output_dir = ROOT / "data" / "interim" / "legacy_hourly_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_interval in TEST_FILES.items():
        source_path = ROOT / "data" / "raw" / filename

        print("\n" + "=" * 76)
        print(f"FILE: {filename}")

        df = pd.read_excel(
            source_path,
            sheet_name="Sheet1",
            header=1,
            usecols=USECOLS,
        )

        df = df.rename(columns={df.columns[0]: "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).copy()
        df = df.sort_values("Timestamp")

        duplicates = int(df["Timestamp"].duplicated().sum())
        if duplicates:
            raise ValueError(
                f"{filename}: found {duplicates} duplicate timestamps."
            )

        intervals = df["Timestamp"].diff().dt.total_seconds().dropna()
        observed_interval = float(intervals.mode().iloc[0])

        if observed_interval != expected_interval:
            raise ValueError(
                f"{filename}: expected {expected_interval}s sampling, "
                f"observed {observed_interval}s."
            )

        missing_columns = set(COLUMN_MAP) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{filename}: expected columns not found: {missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

        df = df.rename(columns=COLUMN_MAP)
        measurement_columns = list(COLUMN_MAP.values())

        for column in measurement_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        indexed = df.set_index("Timestamp")

        hourly = indexed[measurement_columns].resample(
            "1h", closed="left", label="left"
        ).mean()

        # Count timestamped source rows per hour independently of missing values.
        hourly["Source rows per hour"] = (
            indexed["Wind (MW) reported legacy"]
            .resample("1h", closed="left", label="left")
            .size()
        )

        expected_per_hour = int(3600 / expected_interval)
        hourly["Expected rows per hour"] = expected_per_hour
        hourly["Source interval (seconds)"] = expected_interval
        hourly["Source file"] = filename

        incomplete = hourly["Source rows per hour"] != expected_per_hour
        print(f"Source rows: {len(df):,}")
        print(f"Hourly rows: {len(hourly):,}")
        print(f"Incomplete hourly bins: {int(incomplete.sum())}")

        # Relationship checks on the hourly means.
        # These test numerical consistency; they do not establish
        # the physical meaning or accounting boundaries of the columns.
        hourly["Solar+Wind expected from components (MW)"] = (
            hourly["Solar (MW) derived legacy"]
            + hourly["Wind (MW) reported legacy"]
        )
        hourly["Solar+Wind residual (MW)"] = (
            hourly["Solar+Wind (MW) source-derived"]
            - hourly["Solar+Wind expected from components (MW)"]
        )

        hourly["Total minus demand (MW)"] = (
            hourly["Total Generation (MW) source-derived"]
            - hourly["Demand (MW) legacy"]
        )

        # Total may include thermal, hydro, gas, NPC and other components.
        # We cannot validate its full sum with this selected subset alone.
        # Preserve this as a diagnostic, not a claim that total should equal
        # demand or solar + wind.
        numeric_check_columns = [
            "Solar+Wind residual (MW)",
            "Total minus demand (MW)",
        ]

        print("\nSolar + wind consistency:")
        print(
            hourly["Solar+Wind residual (MW)"]
            .describe()
            .round(4)
            .to_string()
        )
        print(
            "Hours with absolute solar+wind residual > 1 MW:",
            int(
                (hourly["Solar+Wind residual (MW)"].abs() > 1).sum()
            ),
        )

        print("\nTotal generation minus demand (diagnostic only):")
        print(
            hourly["Total minus demand (MW)"]
            .describe()
            .round(2)
            .to_string()
        )

        # Check how many measurement values are missing after resampling.
        print("\nMissing hourly measurement values:")
        print(hourly[measurement_columns].isna().sum().to_string())

        output_path = output_dir / (
            source_path.stem.replace(" ", "_").lower()
            + "_hourly_test.csv"
        )
        hourly.reset_index().to_csv(output_path, index=False)

        print(f"\nSaved: {output_path.relative_to(ROOT)}")
        print("First 3 hourly rows:")
        print(hourly.head(3).round(3).to_string())


if __name__ == "__main__":
    main()
