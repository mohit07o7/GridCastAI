
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_grid_data_validated.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "renewable_baseline_metrics.csv"
)

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Explicit chronological split: 70% train, 15% validation, 15% test.
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

# Forecast each measurement 24 hours after the forecast origin.
# At time t, the baseline prediction for t+24 is the value at t.
targets = {
    "Wind (MW)": "Wind (MW)",
    "Solar (MW)": "Solar (MW) corrected",
}

results = []

for label, source_column in targets.items():
    frame = df[["Timestamp", source_column]].copy()
    frame["Actual"] = frame[source_column].shift(-24)
    frame["Prediction"] = frame[source_column]

    # Restrict scoring to forecast targets in the held-out test period.
    test = frame.iloc[validation_end:n - 24].copy()
    test = test.dropna(subset=["Actual", "Prediction"])

    actual = test["Actual"].to_numpy()
    predicted = test["Prediction"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE is unreliable when actual values are zero or near zero.
    nonzero = np.abs(actual) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    results.append({
        "Target": label,
        "Forecast Horizon (hours)": 24,
        "Baseline": "Same-hour-previous-day",
        "Test Observations": len(test),
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "MAPE (%)": mape,
    })

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 72)
print("GRIDCASTAI — 24-HOUR RENEWABLE FORECAST BASELINES")
print("=" * 72)
print(results_df.to_string(index=False))
print(f"\nSaved to: {OUTPUT_PATH}")
print("Raw data was not modified.")
