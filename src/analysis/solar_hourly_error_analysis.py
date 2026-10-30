
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "reports"
    / "experiments"
    / "solar_daytime_experiment_predictions.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "experiments"
    / "solar_hourly_error_analysis.csv"
)


def calculate_metrics(group, prediction_column):
    actual = group["Actual (MW)"].to_numpy(dtype=float)
    predicted = group[prediction_column].to_numpy(dtype=float)

    errors = predicted - actual

    return pd.Series({
        "Observations": len(group),
        "MAE (MW)": np.mean(np.abs(errors)),
        "RMSE (MW)": np.sqrt(np.mean(errors ** 2)),
        "Mean Error (MW)": np.mean(errors),
    })


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "Target Timestamp",
        "Actual (MW)",
        "Seasonal-naive Prediction (MW)",
        "XGBoost Clipped Prediction (MW)",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Target Timestamp"] = pd.to_datetime(
        df["Target Timestamp"],
        errors="raise",
    )

    numeric_columns = required_columns[1:]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in required prediction columns."
        )

    if not np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite numeric values found in predictions."
        )

    if df["Target Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate target timestamps found."
        )

    df["Hour"] = df["Target Timestamp"].dt.hour

    models = {
        "Seasonal-naive": "Seasonal-naive Prediction (MW)",
        "XGBoost": "XGBoost Clipped Prediction (MW)",
    }

    rows = []

    for hour, group in df.groupby("Hour", sort=True):
        for model_name, prediction_column in models.items():
            metrics = calculate_metrics(
                group,
                prediction_column,
            )

            rows.append({
                "Hour": hour,
                "Model": model_name,
                **metrics.to_dict(),
            })

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(OUTPUT_PATH, index=False)

    print("=" * 76)
    print("GRIDCASTAI — HOURLY SOLAR ERROR ANALYSIS")
    print("=" * 76)

    print(f"Input observations: {len(df):,}")
    print(
        "Test period: "
        f"{df['Target Timestamp'].min()} to "
        f"{df['Target Timestamp'].max()}"
    )

    print("\nHOURLY MAE (MW)")
    print(
        results.pivot(
            index="Hour",
            columns="Model",
            values="MAE (MW)",
        ).round(2).to_string()
    )

    print("\nHOURLY RMSE (MW)")
    print(
        results.pivot(
            index="Hour",
            columns="Model",
            values="RMSE (MW)",
        ).round(2).to_string()
    )

    print("\nHOURLY MEAN ERROR / BIAS (MW)")
    print(
        results.pivot(
            index="Hour",
            columns="Model",
            values="Mean Error (MW)",
        ).round(2).to_string()
    )

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
