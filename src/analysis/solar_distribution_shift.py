
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "hourly_grid_data_validated.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "experiments"
    / "solar_distribution_shift.csv"
)

TARGET = "Solar (MW) corrected"


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    required = {"Timestamp", TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if df[TARGET].isna().any():
        raise ValueError("Missing solar values found.")

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Negative corrected solar values found; investigate first."
        )

    # Chronological split by target-row position.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df["Split"] = "Test"
    df.loc[:train_end - 1, "Split"] = "Train"
    df.loc[train_end:val_end - 1, "Split"] = "Validation"

    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.to_period("M").astype(str)

    print("=" * 78)
    print("GRIDCASTAI — SOLAR DISTRIBUTION SHIFT")
    print("=" * 78)

    print("\nCHRONOLOGICAL SPLIT PERIODS")
    for split in ["Train", "Validation", "Test"]:
        part = df[df["Split"] == split]
        print(
            f"{split:12s} | n={len(part):5d} | "
            f"{part['Timestamp'].min()} to "
            f"{part['Timestamp'].max()}"
        )

    print("\nSOLAR DISTRIBUTION BY SPLIT")
    summary = (
        df.groupby("Split", sort=False)[TARGET]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    print(summary.to_string())

    print("\nMEAN SOLAR GENERATION BY HOUR AND SPLIT (MW)")
    hourly = df.pivot_table(
        index="Hour",
        columns="Split",
        values=TARGET,
        aggfunc="mean",
    ).round(2)
    print(hourly.to_string())

    print("\nMONTHLY MEAN SOLAR GENERATION (MW)")
    monthly = df.pivot_table(
        index="Month",
        columns="Split",
        values=TARGET,
        aggfunc="mean",
    ).round(2)
    print(monthly.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    hourly.to_csv(OUTPUT_PATH)
    print(f"\nHourly comparison saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
