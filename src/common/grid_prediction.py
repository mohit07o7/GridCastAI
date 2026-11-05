"""Load-only, origin-safe demand/wind/solar prediction service."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from common.demand_model import load_demand_bundle, sha256
from common.inference_history import (
    DEMAND_COLUMNS, WIND_COLUMNS, HOUR, POLICY_VERSION, HistoryError,
    prepare_history, parse_origin, recent_index, demand_features, wind_features,
    quality_flags, check_quality,
)

ROOT = Path(__file__).resolve().parents[2]


def _verified_file(root, spec):
    path = root / spec['path']
    if sha256(path) != spec['sha256']:
        raise ValueError(f'Artifact hash mismatch: {spec["path"]}')
    return path


def load_wind_model(root, entry):
    path = _verified_file(root, entry['artifact'])
    _verified_file(root, entry['metadata'])
    model = XGBRegressor()
    model.load_model(path)
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    if (entry['features'] != WIND_COLUMNS or booster.feature_names != WIND_COLUMNS
            or booster.num_features() != len(WIND_COLUMNS)
            or booster.num_boosted_rounds() != entry['boosted_rounds']
            or config['learner']['objective']['name'] != 'reg:absoluteerror'
            or entry['objective'] != 'reg:absoluteerror'):
        raise ValueError('Wind model structure does not match selected v3 definition')
    metadata = json.loads((root / entry['metadata']['path']).read_text())['targets']['wind']
    if (metadata['feature_columns'] != WIND_COLUMNS
            or metadata['best_iteration'] + 1 != entry['boosted_rounds']
            or metadata['last_validation_target'] != entry['latest_training_or_tuning_target']
            or metadata['selection']['all_hours'] != 'residual_xgb'):
        raise ValueError('Wind selection metadata mismatch')
    return model


class GridForecaster:
    """Cache validated artifacts. forecast(history, origin) never fits a model."""
    def __init__(self, root=ROOT, manifest=None):
        self.root = Path(root)
        manifest_path = Path(manifest) if manifest else self.root / 'models/forecast_manifest.json'
        try:
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest['schema_version'] != 1 or self.manifest['horizon_hours'] != 24:
                raise ValueError('Unsupported forecast manifest')
            if self.manifest['inference_quality_policy'] != POLICY_VERSION:
                raise ValueError('Unsupported inference quality policy')
            self.entries = self.manifest['components']
            for component in ['demand', 'wind', 'solar']:
                entry = self.entries[component]
                if not isinstance(entry, dict) or not isinstance(entry.get('id'), str):
                    raise ValueError('Invalid component manifest')
                if component != 'solar' and not all(isinstance(entry.get(key), dict) for key in ['artifact', 'metadata']):
                    raise ValueError('Invalid component artifact specification')
            if self.entries['demand']['method'] != 'xgboost' or self.entries['wind']['method'] != 'residual_xgb':
                raise ValueError('Unsupported selected model methods')
            if self.entries['solar']['method'] != 'prior_day_persistence':
                raise ValueError('Only selected solar persistence is supported')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HistoryError(f'Cannot read forecast manifest: {exc}') from exc
        self.models = {}
        self.artifact_errors = {}
        for component in ['demand', 'wind']:
            try:
                entry = self.entries[component]
                if component == 'demand':
                    _verified_file(self.root, entry['artifact'])
                    metadata_path = _verified_file(self.root, entry['metadata'])
                    model = load_demand_bundle(metadata_path.parent)
                    if ([f['name'] for f in model.metadata['features']] != DEMAND_COLUMNS
                            or entry['features'] != DEMAND_COLUMNS
                            or model.metadata['splits']['validation']['last_target'] != entry['latest_training_or_tuning_target']):
                        raise ValueError('Demand manifest feature/cutoff mismatch')
                else:
                    model = load_wind_model(self.root, entry)
                parse_origin(entry['latest_training_or_tuning_target'])
                self.models[component] = model
            except (OSError, ValueError, KeyError, TypeError, XGBoostError) as exc:
                self.artifact_errors[component] = f'{component}:artifact_unavailable ({exc})'

    def forecast(self, history, origin):
        origin = parse_origin(origin)
        past = prepare_history(history, origin)
        flags, zeros_available, balance_available = quality_flags(past)
        recent = recent_index(origin)
        annual = origin - 8736 * HOUR
        # Report quality checks whose required past context is unavailable.
        hourly_step = past.index.to_series().diff().eq(HOUR)
        contiguous_six = hourly_step.rolling(5, min_periods=5).sum().eq(5)
        six_hour_context = {component: contiguous_six & np.isfinite(past[f'{component}_mw']).rolling(6, min_periods=6).sum().eq(6)
                            for component in ['demand', 'wind', 'solar']}
        output = {}
        for component in ['demand', 'wind', 'solar']:
            entry = self.entries[component]
            required = {'demand': recent, 'wind': recent.union(pd.DatetimeIndex([annual])),
                        'solar': pd.DatetimeIndex([origin])}[component]
            reasons = check_quality(flags, required, component)
            quality_checks = [(component, required)]
            if component == 'demand':
                lag_times = pd.DatetimeIndex([origin-HOUR, origin-24*HOUR])
                for exogenous in ['wind', 'solar']:
                    reasons += check_quality(flags, lag_times, exogenous)
                    quality_checks.append((exogenous, lag_times))
            unavailable = []
            for checked, times in quality_checks:
                available_times = times.intersection(past.index)
                if len(available_times) < len(times):
                    unavailable.append(f'{checked}:quality_checks_at_missing_hours')
                if not six_hour_context[checked].reindex(times, fill_value=False).all():
                    unavailable.append(f'{checked}:six_hour_flatline_context_incomplete')
                if checked == 'demand':
                    previous_times = times - HOUR
                    if not previous_times.isin(past.index).all():
                        unavailable.append('demand:sharp_change_previous_hour_missing')
                if not zeros_available.reindex(times, fill_value=False).all():
                    unavailable.append(f'{checked}:all_zero_check_incomplete')
                if checked in ['wind', 'solar'] and not balance_available.reindex(times, fill_value=False).all():
                    unavailable.append(f'{checked}:renewable_component_balance_check_incomplete')
            origin_reasons = check_quality(flags, [origin], component)
            method = entry['method']
            forecast = None
            status = 'ok'
            if component != 'solar':
                if component in self.artifact_errors:
                    reasons.append(self.artifact_errors[component])
                elif origin < parse_origin(entry['latest_training_or_tuning_target']):
                    reasons.append(f'{component}:model_training_or_tuning_after_origin')
                if not reasons:
                    try:
                        features = demand_features(past, origin) if component == 'demand' else wind_features(past, origin)
                        if not np.isfinite(features.to_numpy(dtype=float)).all():
                            raise ValueError('nonfinite_features')
                        if component == 'demand':
                            forecast = float(self.models[component].predict(features)[0])
                        else:
                            model = self.models[component]
                            delta = float(model.predict(features, iteration_range=(0, entry['boosted_rounds']))[0])
                            forecast = float(past.at[origin, 'wind_mw']) + delta
                        if not np.isfinite(forecast) or forecast < 0:
                            raise ValueError('invalid_model_output')
                    except (ValueError, KeyError, XGBoostError) as exc:
                        reasons.append(f'{component}:prediction_unavailable ({exc})')
                        forecast = None
            elif not reasons:
                forecast = float(past.at[origin, 'solar_mw'])
            if forecast is None:
                if not origin_reasons:
                    forecast = float(past.at[origin, f'{component}_mw'])
                    method = 'prior_day_persistence'
                    status = 'fallback' if component != 'solar' else 'ok'
                else:
                    status, method = 'blocked', None
                    reasons += origin_reasons
            output[component] = {
                'target_timestamp': str(origin + 24*HOUR), 'forecast_mw': forecast,
                'method': method, 'status': status, 'reasons': sorted(set(reasons)),
                'unavailable_quality_checks': sorted(set(unavailable)),
                'selected_model_id': entry['id'],
                'selected_artifact_sha256': entry.get('artifact', {}).get('sha256'),
                'artifact_validated': component in self.models if component != 'solar' else None,
            }
        statuses = [c['status'] for c in output.values()]
        overall = 'blocked' if all(s == 'blocked' for s in statuses) else ('ok' if all(s == 'ok' for s in statuses) else 'degraded')
        return {
            'schema_version': 1, 'origin': str(origin), 'target_timestamp': str(origin+24*HOUR),
            'horizon_hours': 24, 'timestamp_convention': 'timezone-naive source clock; no conversion',
            'origin_observation_assumed_available': True,
            'run_context': 'retrospective_simulation_with_fixed_saved_models; not evidence of an actually issued forecast',
            'inference_quality_policy': POLICY_VERSION, 'target_quality': 'unknown',
            'status': overall, 'components': output,
        }


def predict_grid(history, origin, root=ROOT):
    """Convenience API; reuse GridForecaster for repeated requests."""
    return GridForecaster(root=root).forecast(history, origin)
