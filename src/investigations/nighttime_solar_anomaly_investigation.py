
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "hourly_grid_data_validated.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "investigations"
    / "nighttime_solar_anomaly_context.csv"
)

SOLAR = "Solar (MW) corrected"
REPORTED = "Solar (MW) reported"

ENERGY_COLUMNS = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW) reported",
    "Solar (MW) corrected",
    "Total Generation (MW)",
]


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    required = ["Timestamp", SOLAR, REPORTED]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    df["Hour"] = df["Timestamp"].dt.hour

    # Flag nighttime observations with unusually high solar values.
    flagged_mask = (
        df["Hour"].between(0, 5)
        & (pd.to_numeric(df[SOLAR], errors="coerce") > 1000)
    )

    flagged_positions = set(df.index[flagged_mask])

    # Include three observations before and after each flagged row.
    context_positions = set()

    for position in flagged_positions:
        start = max(0, position - 3)
        end = min(len(df) - 1, position + 3)
        context_positions.update(range(start, end + 1))

    context = df.loc[sorted(context_positions)].copy()

    context["Flagged_Nighttime_Solar"] = context.index.isin(
        flagged_positions
    )

    context["Hours_From_Flagged_Record"] = pd.NA

    for position in flagged_positions:
        nearby = context.index[
            (context.index >= position - 3)
            & (context.index <= position + 3)
        ]

        context.loc[nearby, "Hours_From_Flagged_Record"] = [
            index - position for index in nearby
        ]

    output_columns = [
        "Timestamp",
        "Hour",
        "Flagged_Nighttime_Solar",
        "Hours_From_Flagged_Record",
    ] + [col for col in ENERGY_COLUMNS if col in context.columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    context[output_columns].to_csv(OUTPUT_PATH, index=False)

    print("=" * 78)
    print("GRIDCASTAI — NIGHTTIME SOLAR ANOMALY CONTEXT")
    print("=" * 78)
    print(f"Source rows: {len(df):,}")
    print(f"Flagged nighttime rows: {len(flagged_positions):,}")
    print(f"Rows in context report: {len(context):,}")

    print("\nFLAGGED RECORDS — ORIGINAL AND CORRECTED SOLAR")
    flagged = df.loc[
        sorted(flagged_positions),
        ["Timestamp", "Demand (MW)", "Wind (MW)",
         REPORTED, SOLAR, "Total Generation (MW)"],
    ]

    print(flagged.to_string(index=False))

    print("\nMARCH 2025 NIGHTTIME VALUES ABOVE 1000 MW")
    march = df[
        (df["Timestamp"].dt.to_period("M") == "2025-03")
        & df["Hour"].between(0, 5)
        & (df[SOLAR] > 1000)
    ]

    print(march[
        ["Timestamp", "Demand (MW)", "Wind (MW)",
         REPORTED, SOLAR, "Total Generation (MW)"]
    ].to_string(index=False))

    print(f"\nContext report saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
