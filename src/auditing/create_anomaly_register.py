
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
output_path = PROJECT_ROOT / "reports" / "audits" / "anomaly_register.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

anomalies = [
    {
        "anomaly_id": "A001",
        "start_timestamp": "2024-06-06 18:00:00",
        "end_timestamp": "2024-06-10 08:00:00",
        "affected_columns": "Total Generation (MW); Wind (MW)",
        "observed_pattern": (
            "Total Generation remains 212887.48 MW and "
            "Wind remains 11065.84 MW for 87 hourly observations"
        ),
        "severity": "High",
        "status": "Flagged - cause unconfirmed",
        "action": "Preserve original values; exclude Total Generation from trusted model targets pending investigation",
    },
    {
        "anomaly_id": "A002",
        "start_timestamp": "2025-04-04 16:00:00",
        "end_timestamp": "2025-04-04 16:00:00",
        "affected_columns": "Demand (MW); Total Generation (MW)",
        "observed_pattern": (
            "Demand falls to 97065.28 MW between "
            "203399.28 MW at 15:00 and 200036.27 MW at 17:00"
        ),
        "severity": "High",
        "status": "Flagged - cause unconfirmed",
        "action": "Preserve observation; investigate sensitivity of forecasts to this point",
    },
]

pd.DataFrame(anomalies).to_csv(output_path, index=False)

print(f"Anomaly register saved to: {output_path}")
print(pd.DataFrame(anomalies).to_string(index=False))
