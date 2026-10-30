
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January 2024- June 2025.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "january_2025_flatline_context.csv"
)

df = pd.read_excel(SOURCE_PATH, sheet_name="Report")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce",
)

columns = [
    "Demand (MW)",
    "Wind (MW)",
    "Solar (MW)",
    "Total Generation (MW)",
]

for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = (
    df.dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

# Inspect the two January flatline intervals together.
start = pd.Timestamp("2025-01-07 20:00:00")
end = pd.Timestamp("2025-01-11 05:00:00")

window = df.loc[
    df["Timestamp"].between(start, end),
    ["Timestamp", *columns],
].copy()

# Mark the previously identified repeated-value intervals.
window["Inside_Jan_08_09_Run"] = (
    window["Timestamp"].between(
        pd.Timestamp("2025-01-08 20:00:00"),
        pd.Timestamp("2025-01-09 06:00:00"),
    )
)

window["Inside_Jan_09_10_Run"] = (
    window["Timestamp"].between(
        pd.Timestamp("2025-01-09 23:00:00"),
        pd.Timestamp("2025-01-10 05:00:00"),
    )
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
window.to_csv(OUTPUT_PATH, index=False)

print("=" * 72)
print("GRIDCASTAI — JANUARY 2025 FLATLINE CONTEXT")
print("=" * 72)
print(f"Rows extracted: {len(window)}")
print(f"Time range: {start} to {end}")
print()
print(window.to_string(index=False))
print()
print(f"Saved to: {OUTPUT_PATH}")
print("The raw workbook was not modified.")
