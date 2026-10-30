
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "reports" / "experiments"
    / "solar_daytime_experiment_predictions.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "experiments"
    / "solar_prediction_profile_analysis.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    df["Target Timestamp"] = pd.to_datetime(
        df["Target Timestamp"], errors="raise"
    )

    df["Hour"] = df["Target Timestamp"].dt.hour

    actual_col = "Actual (MW)"
    naive_col = "Seasonal-naive Prediction (MW)"
    xgb_col = "XGBoost Clipped Prediction (MW)"

    profile = (
        df.groupby("Hour")
        .agg(
            Observations=(actual_col, "size"),
            Actual_Mean_MW=(actual_col, "mean"),
            Seasonal_Naive_Mean_MW=(naive_col, "mean"),
            XGBoost_Mean_MW=(xgb_col, "mean"),
        )
        .reset_index()
    )

    profile["Baseline_Bias_MW"] = (
        profile["Seasonal_Naive_Mean_MW"]
        - profile["Actual_Mean_MW"]
    )

    profile["XGBoost_Bias_MW"] = (
        profile["XGBoost_Mean_MW"]
        - profile["Actual_Mean_MW"]
    )

    profile["XGBoost_Underprediction_Pct"] = (
        100
        * (
            profile["Actual_Mean_MW"]
            - profile["XGBoost_Mean_MW"]
        )
        / profile["Actual_Mean_MW"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    profile.to_csv(OUTPUT_PATH, index=False)

    display_columns = [
        "Hour",
        "Observations",
        "Actual_Mean_MW",
        "Seasonal_Naive_Mean_MW",
        "XGBoost_Mean_MW",
        "Baseline_Bias_MW",
        "XGBoost_Bias_MW",
        "XGBoost_Underprediction_Pct",
    ]

    print("=" * 88)
    print("GRIDCASTAI — SOLAR PREDICTION PROFILE")
    print("=" * 88)
    print(profile[display_columns].round(2).to_string(index=False))
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
