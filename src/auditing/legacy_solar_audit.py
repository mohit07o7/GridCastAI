
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "reports" / "audits" / "legacy_solar_clipping_audit.csv"

results = []

files = sorted(RAW_DIR.glob("*.xlsx"))

for i, file in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] {file.name}", flush=True)

    try:
        # Read only the two solar columns; no need to open ExcelFile separately.
        df = pd.read_excel(
            file,
            sheet_name="Sheet1",
            header=1,
            usecols="G:H",
            engine="openpyxl",
        )

        raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        derived = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        valid = raw.notna() & derived.notna()
        mismatch = (derived - raw).abs() > 0.01

        results.append({
            "file": file.name,
            "rows": int(valid.sum()),
            "negative_raw_solar": int((valid & (raw < 0)).sum()),
            "mismatches": int((valid & mismatch).sum()),
            "negative_clipped_to_zero": int(
                (valid & (raw < 0) & (derived == 0)).sum()
            ),
            "other_mismatches": int(
                (valid & mismatch & (raw >= 0)).sum()
            ),
            "max_clipping_error": (
                float(
                    (derived - raw.clip(lower=0)).abs()[valid].max()
                )
                if valid.any()
                else None
            ),
        })

    except Exception as exc:
        print(f"  Skipped: {exc}", flush=True)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUT_FILE, index=False)

print(f"\nAudit complete: {len(results)} files processed.")
print(f"Report saved to: {OUT_FILE}")
