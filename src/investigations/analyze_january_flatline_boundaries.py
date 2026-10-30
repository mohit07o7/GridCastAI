
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January 2024- June 2025.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "january_flatline_boundary_analysis.csv"
)

MEASUREMENTS = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

df = pd.read_excel(INPUT_PATH, sheet_name="Report")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce",
)

for column in MEASUREMENTS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = (
    df.dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

df["Demand minus total generation (MW)"] = (
    df["Demand (MW)"] - df["Total Generation (MW)"]
)

# These timestamps mark the first and last rows of each flatline.
boundaries = [
    ("Run 1 start", "2025-01-08 20:00:00"),
    ("Run 1 end", "2025-01-09 06:00:00"),
    ("Run 2 start", "2025-01-09 23:00:00"),
    ("Run 2 end", "2025-01-10 05:00:00"),
]

results = []

for label, timestamp_text in boundaries:
    timestamp = pd.Timestamp(timestamp_text)

    for offset, position in [
        ("Previous hour", timestamp - pd.Timedelta(hours=1)),
        ("Boundary hour", timestamp),
        ("Next hour", timestamp + pd.Timedelta(hours=1)),
    ]:
        row = df.loc[df["Timestamp"].eq(position)]

        if row.empty:
            results.append({
                "Boundary": label,
                "Position": offset,
                "Timestamp": position,
                "Row found": False,
            })
            continue

        record = row.iloc[0].to_dict()
        record["Boundary"] = label
        record["Position"] = offset
        record["Row found"] = True
        results.append(record)

result_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

display_columns = [
    "Boundary",
    "Position",
    "Timestamp",
    *MEASUREMENTS,
    "Demand minus total generation (MW)",
]

print("=" * 76)
print("JANUARY 2025 FLATLINE BOUNDARY ANALYSIS")
print("=" * 76)
print(result_df[display_columns].to_string(index=False))
print(f"\nSaved to: {OUTPUT_PATH}")
print("Original workbook was not modified.")
