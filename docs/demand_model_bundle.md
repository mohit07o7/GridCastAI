# Demand v1 model bundle

This bundle preserves the evaluated main 13-feature demand model. It is fitted on training rows only, with validation early stopping, and predicts demand 24 hours after each origin. It is not a refit on all available data and is not the trend experiment. The test period is a retrospective benchmark, not a new untouched holdout.

From `/Users/mohityadav/Desktop/GridCastAI`, train once and save:

```bash
.venv/bin/python src/forecasting/xgboost_demand_forecast.py
```

The script first checks predictions against the existing saved experiment (0.02 MW CSV tolerance). A mismatch fails before replacing outputs. It writes `models/demand_v1/model.json`, `metadata.json`, `test_reference.csv`, and `training_reference.json`. The reference preserves the original in-memory float32 predictions as float64 CSV values for exact replay. Do not edit the bundle files independently.

Metadata records ordered feature names/types, model parameters, selected iteration/range, policy version, actual split row counts and origin/target dates, versions, source/code hashes, and metrics. A model attribute binds the metadata to the fitted model; metadata also records the model-file hash. These checks detect accidental changes or mismatched files; they are not a security signature. Files are validated in staging before replacement; an interrupted two-file replacement fails hash validation on load.

Verify in a new process without training:

```bash
.venv/bin/python src/investigations/verify_demand_model.py \
  --output reports/audits/demand_model_bundle_verification.json
.venv/bin/python -m unittest discover -s tests
```

The independent verifier disables XGBoost fit/train/update entry points and never imports the training script. It checks code/data hashes, split reconstruction and one-to-one timestamps, then compares loaded predictions against the full-precision training reference at absolute tolerance 0.000001 MW and the experiment CSV at 0.02 MW. It recomputes MAE, RMSE and MAPE. January 10, 2025 15:00 is excluded from scoring due to frozen demand in its input history; its printed prediction is diagnostic only.

Reusable feature-level inference:

```python
from common.demand_model import load_demand_bundle
predictor = load_demand_bundle('models/demand_v1')
predictions_mw = predictor.predict(feature_dataframe)
```

Run Python with `src` on its import path, for example `PYTHONPATH=src .venv/bin/python your_script.py`. The loader orders required columns by saved names and permits extra columns. It rejects missing/duplicate columns, empty input, strings, booleans, complex numbers, missing and infinite values. Numeric precision may differ from recorded training dtypes; real numeric integer/float inputs are accepted. Prediction explicitly uses `[0, best_iteration + 1)` boosting rounds.

The loader requires neither the original data nor the training script. The retrospective verifier requires both to validate provenance. Exact library versions are recorded; use the tested pinned environment for reproducibility. Native JSON loading does not promise cross-version bitwise equality. Live history ingestion, online quality handling, and production model selection are later tasks.
