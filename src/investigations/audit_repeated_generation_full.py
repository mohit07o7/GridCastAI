"""
GridCastAI — Full Dataset Repeated Generation Audit
Audits long runs of consecutive repeated values in Wind and Total Generation
across the entire processed dataset.

Outputs:
- reports/audits/full_dataset_repeated_runs.csv
- reports/audits/full_dataset_repeated_runs_summary.csv

Diagnostic investigation script: does not modify or correct any source observations.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "audits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_PATH = OUTPUT_DIR / "full_dataset_repeated_runs.csv"
SUMMARY_PATH = OUTPUT_DIR / "full_dataset_repeated_runs_summary.csv"

print("=" * 70)
print("GRIDCASTAI — FULL DATASET REPEATED GENERATION AUDIT")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Validated dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

print(f"\nDataset loaded: {len(df):,} rows")
print(f"Time span: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

target_columns = ["Wind (MW)", "Total Generation (MW)"]
available_targets = [c for c in target_columns if c in df.columns]

if not available_targets:
    raise ValueError(f"None of {target_columns} found in dataset columns: {df.columns.tolist()}")

def find_repeated_runs(series, timestamps, min_hours=6):
    """Identify consecutive runs of identical non-null values."""
    is_diff = series.ne(series.shift())
    run_id = is_diff.cumsum()
    
    runs = (
        pd.DataFrame({
            "Timestamp": timestamps,
            "Value": series,
            "RunID": run_id,
        })
        .groupby("RunID", sort=False)
        .agg(
            Start=("Timestamp", "first"),
            End=("Timestamp", "last"),
            Hours=("Timestamp", "size"),
            Value=("Value", "first"),
        )
        .reset_index(drop=True)
    )
    return runs[runs["Hours"] >= min_hours].sort_values("Hours", ascending=False).reset_index(drop=True)

all_runs_list = []
summary_list = []

for col in available_targets:
    runs = find_repeated_runs(df[col], df["Timestamp"], min_hours=6)
    runs.insert(0, "Column", col)
    all_runs_list.append(runs)
    
    total_runs = len(runs)
    total_hours_affected = runs["Hours"].sum() if total_runs > 0 else 0
    max_duration = runs["Hours"].max() if total_runs > 0 else 0
    
    summary_list.append({
        "Column": col,
        "Runs (>=6h)": total_runs,
        "Max Consecutive Hours": max_duration,
        "Total Hours Affected": total_hours_affected,
        "Pct of Total Dataset": round((total_hours_affected / len(df)) * 100, 2),
    })
    
    print(f"\n{col}:")
    print(f"  Total runs (>= 6 hours): {total_runs}")
    print(f"  Max consecutive duration: {max_duration} hours")
    print(f"  Total hours affected: {total_hours_affected} ({round((total_hours_affected / len(df)) * 100, 2)}%)")
    if not runs.empty:
        print("  Top 5 longest repeated runs:")
        print(runs.head(5)[["Start", "End", "Hours", "Value"]].to_string(index=False))

# Combine and save runs
if all_runs_list:
    combined_runs = pd.concat(all_runs_list, ignore_index=True)
    combined_runs.to_csv(RUNS_PATH, index=False)
    print(f"\nSaved detailed repeated runs to: {RUNS_PATH}")

# Save summary table
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"Saved repeated runs summary to: {SUMMARY_PATH}")

print("\n" + summary_df.to_string(index=False))
print("\nAudit complete. No source observations were modified.")
