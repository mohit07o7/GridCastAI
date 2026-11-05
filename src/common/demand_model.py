"""Native XGBoost demand artifacts; importing this module never trains a model."""
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_complex_dtype
from xgboost import XGBRegressor


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fingerprint(metadata):
    # Bind metadata to a model attribute, excluding the resulting model-file hash.
    content = {k: v for k, v in metadata.items() if k != 'model_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _range(frame):
    return {'rows': len(frame), 'first_origin': str(frame.Timestamp.min()),
            'last_origin': str(frame.Timestamp.max()),
            'first_target': str(frame['Target Timestamp'].min()),
            'last_target': str(frame['Target Timestamp'].max())}


def save_demand_bundle(model, bundle_dir, features, splits, data_path, code_paths,
                       project_root, metrics):
    """Save the fitted training-only estimator, metadata, and provenance."""
    root = Path(project_root)
    training = splits['train']
    metadata = {
        'schema_version': 1, 'model_id': 'demand_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'target': 'demand_mw', 'forecast_horizon_hours': 24,
        'timestamp_convention': 'timezone-naive source clock; no conversion',
        'features': [{'name': c, 'dtype': str(training[c].dtype)} for c in features],
        'quality_policy_version': 'demand_quality_v1',
        'fit_scope': 'training only; validation used for early stopping; no refit',
        'parameters': _json_safe(model.get_params()),
        'best_iteration': int(model.best_iteration),
        'prediction_iteration_range': [0, int(model.best_iteration) + 1],
        'boosted_rounds': model.get_booster().num_boosted_rounds(),
        'splits': {name: _range(frame) for name, frame in splits.items()},
        'versions': {'python': platform.python_version(), **{n: version(n) for n in
                     ['numpy', 'pandas', 'xgboost', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']}},
        'source_data': {'path': str(Path(data_path).relative_to(root)), 'sha256': sha256(data_path)},
        'code_sha256': {str(Path(p).relative_to(root)): sha256(p) for p in code_paths},
        'test_metrics': metrics,
    }
    model.get_booster().set_attr(gridcast_metadata_sha256=_fingerprint(metadata))
    destination = Path(bundle_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as staging:
        staging = Path(staging)
        model.save_model(staging / 'model.json')
        metadata['model_sha256'] = sha256(staging / 'model.json')
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        load_demand_bundle(staging)
        (staging / 'model.json').replace(destination / 'model.json')
        (staging / 'metadata.json').replace(destination / 'metadata.json')
    return metadata


@dataclass(frozen=True)
class DemandPredictor:
    model: XGBRegressor
    metadata: dict

    def predict(self, frame):
        """Return MW predictions; select named columns in recorded order."""
        if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
            raise ValueError('Expected a DataFrame with unique column names')
        names = [f['name'] for f in self.metadata['features']]
        missing = [c for c in names if c not in frame]
        if missing:
            raise ValueError(f'Missing features: {missing}')
        selected = frame.loc[:, names]
        for c in names:
            dtype = selected[c].dtype
            if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                raise ValueError(f'Feature {c} must be real numeric data')
        if not np.isfinite(selected.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError('Features must be finite and nonmissing')
        if len(selected) == 0:
            raise ValueError('Expected at least one feature row')
        result = self.model.predict(selected, iteration_range=tuple(self.metadata['prediction_iteration_range']))
        if not np.isfinite(result).all():
            raise ValueError('Model produced nonfinite predictions')
        return result


def load_demand_bundle(bundle_dir):
    """Validate and load model + metadata. Does not import training code or fit."""
    path = Path(bundle_dir)
    try:
        metadata = json.loads((path / 'metadata.json').read_text(),
                              parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f'Invalid JSON constant {v}')))
        if metadata['schema_version'] != 1 or metadata['model_id'] != 'demand_v1':
            raise ValueError('Unsupported demand bundle schema/model')
        if metadata['forecast_horizon_hours'] != 24 or metadata['quality_policy_version'] != 'demand_quality_v1':
            raise ValueError('Unsupported horizon/quality policy')
        features = metadata['features']
        names = [f['name'] for f in features]
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
            raise ValueError('Invalid feature schema')
        for feature in features:
            dtype = np.dtype(feature['dtype'])
            if dtype.kind not in 'iuf':
                raise ValueError('Invalid saved feature dtype')
        if sha256(path / 'model.json') != metadata['model_sha256']:
            raise ValueError('Model file hash does not match metadata')
        model = XGBRegressor()
        model.load_model(path / 'model.json')
        booster = model.get_booster()
        if booster.attr('gridcast_metadata_sha256') != _fingerprint(metadata):
            raise ValueError('Metadata does not match the saved model')
        best = metadata['best_iteration']
        if type(best) is not int or best != model.best_iteration or not 0 <= best < booster.num_boosted_rounds():
            raise ValueError('Invalid selected boosting iteration')
        if metadata['prediction_iteration_range'] != [0, best + 1]:
            raise ValueError('Invalid prediction iteration range')
        if booster.num_boosted_rounds() != metadata['boosted_rounds'] or booster.feature_names != names:
            raise ValueError('Model structure/feature order mismatch')
        if booster.num_features() != len(features):
            raise ValueError('Model feature count mismatch')
        objective = json.loads(booster.save_config())['learner']['objective']['name']
        if objective != metadata['parameters']['objective']:
            raise ValueError('Objective does not match metadata')
        return DemandPredictor(model, metadata)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid demand bundle metadata: {exc}') from exc
