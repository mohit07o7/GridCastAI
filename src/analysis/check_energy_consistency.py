
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "processed" / "hourly_grid_data_validated.csv"
df = pd.read_csv(path, parse_dates=["Timestamp"])

# Use the provisional corrected solar column
df["Calculated Demand Minus Renewables (MW)"] = (
    df["Demand (MW)"]
    - df["Solar (MW) corrected"]
    - df["Wind (MW)"]
)

# Compare demand and reported total generation
df["Generation Minus Demand (MW)"] = (
    df["Total Generation (MW)"] - df["Demand (MW)"]
)

print("DEMAND MINUS RENEWABLE GENERATION")
print(df["Calculated Demand Minus Renewables (MW)"].describe())

print("\nTOTAL GENERATION MINUS DEMAND")
print(df["Generation Minus Demand (MW)"].describe())

print("\nCORRELATION MATRIX")
print(
    df[
        [
            "Demand (MW)",
            "Wind (MW)",
            "Solar (MW) corrected",
            "Total Generation (MW)"
        ]
    ].corr().round(3)
)

print("\nFIRST 10 RECORDS")
print(
    df[
        [
            "Timestamp",
            "Demand (MW)",
            "Wind (MW)",
            "Solar (MW) corrected",
            "Total Generation (MW)",
            "Calculated Demand Minus Renewables (MW)"
        ]
    ].head(10).to_string(index=False)
)
