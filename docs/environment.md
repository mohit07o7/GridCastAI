# Tested environment

CPython 3.14.3, existing Desktop `.venv`, macOS. `.python-version` records the interpreter. `requirements.txt` pins reviewed runtime libraries and numerical dependencies; `requirements-dev.txt` holds optional notebook tools. This is a curated requirements set, not a dump of the whole environment. Full transitive notebook dependencies are not locked. Use `python -m pip install -r requirements.txt` for runtime setup.

All four demand experiments were rerun sequentially in this environment. All XGBoost constructor sites in src were inspected: seed 42 is explicit or inherited from renewable MODEL_PARAMS. The renewable bootstrap uses default_rng(42). Older renewable scripts use n_jobs=-1; they remain historical experiments. See reports/audits/training_randomness_review.json. Cross-version/platform bitwise equality is not promised. Validate with the saved prediction replay, tolerance 0.02 MW for CSV formatting.
