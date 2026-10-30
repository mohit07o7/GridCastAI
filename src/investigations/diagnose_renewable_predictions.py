
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports/experiments/renewable_corrected_split_predictions.csv"
OUTPUT = ROOT / "reports/experiments/renewable_error_diagnostics.csv"

df = pd.read_csv(INPUT, parse_dates=["Target Timestamp"])

rows = []

for target, group in df.groupby("Target"):
    group = group.copy()
    actual = group["Actual (MW)"]
    timestamp = group["Target Timestamp"]

    models = {
        "Seasonal-naive": group["Seasonal-naive Prediction (MW)"],
        "XGBoost": group["XGBoost Prediction (MW)"],
    }

    # These are clock-hour groups, not verified sunrise/sunset periods.
    group["Hour"] = timestamp.dt.hour

    for model_name, prediction in models.items():
        group["Prediction"] = prediction
        group["Error"] = prediction - actual
        group["Absolute Error"] = group["Error"].abs()

        segments = [("All hours", group)]

        if target == "Solar (MW)":
            segments.extend([
                ("Night proxy (00-05, 19-23)", group[
                    group["Hour"].isin([0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23])
                ]),
                ("Day proxy (06-18)", group[
                    group["Hour"].between(6, 18)
                ]),
            ])

        for segment_name, segment in segments:
            if segment.empty:
                continue

            y = segment["Actual (MW)"].to_numpy()
            p = segment["Prediction"].to_numpy()

            rows.append({
                "Target": target,
                "Model": model_name,
                "Segment": segment_name,
                "Observations": len(segment),
                "MAE (MW)": mean_absolute_error(y, p),
                "RMSE (MW)": np.sqrt(mean_squared_error(y, p)),
                "Mean Error (MW)": np.mean(p - y),
                "Negative Predictions": int((p < 0).sum()),
                "Minimum Prediction (MW)": np.min(p),
                "Maximum Prediction (MW)": np.max(p),
            })

result = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("=" * 72)
print("GRIDCASTAI — RENEWABLE PREDICTION DIAGNOSTICS")
print("=" * 72)
print(result.round(3).to_string(index=False))
print("\nSaved to:", OUTPUT)
print("Original predictions and source data were not modified.")
