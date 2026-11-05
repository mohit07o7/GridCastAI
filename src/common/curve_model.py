"""Load-only direct forecast curves. This module never imports training code."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common.curve_features import (BASELINES, COMPONENTS, FEATURES, FEATURE_VERSION,
                                   POLICY, HOUR, CurveHistory, baseline_offset)
from common.inference_history import parse_origin

ROOT = Path(__file__).resolve().parents[2]
METHOD = 'residual_xgboost'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(entry):
    value = {k: v for k, v in entry.items() if k != 'model_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def predict_candidate(model, frame, entry):
    if list(frame.columns) != FEATURES or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Invalid ordered curve model features')
    residual = model.predict(frame, iteration_range=(0, entry['best_iteration']+1))
    return np.maximum(0, frame.target_previous_day.to_numpy()+residual.astype(float))


def load_candidate(directory, entry):
    path = Path(directory)/entry['model_file']
    if path.parent.resolve() != Path(directory).resolve() or digest(path) != entry['model_sha256']:
        raise ValueError('Curve artifact path/hash mismatch')
    model = XGBRegressor()
    model.load_model(path)
    booster = model.get_booster()
    objective = json.loads(booster.save_config())['learner']['objective']['name']
    if (booster.feature_names != FEATURES or booster.num_features() != len(FEATURES)
            or booster.num_boosted_rounds() != entry['boosted_rounds']
            or model.best_iteration != entry['best_iteration']
            or not 0 <= entry['best_iteration'] < entry['boosted_rounds']
            or objective != 'reg:absoluteerror'
            or booster.attr('gridcast_curve_metadata_sha256') != fingerprint(entry)):
        raise ValueError('Curve model structure/metadata mismatch')
    return model


def _reason_lists(masks):
    names = list(masks)
    return [[name for name, active in zip(names, row) if active]
            for row in masks.to_numpy(dtype=bool)]


class GridCurveForecaster:
    def __init__(self, root=ROOT, bundle=None, strict=False):
        self.root = Path(root)
        self.directory = Path(bundle) if bundle is not None else self.root/'models/grid_curve_v1'
        self.manifest = json.loads((self.directory/'metadata.json').read_text())
        meta = self.manifest
        if (meta['schema_version'] != 1 or meta['model_id'] != 'grid_curve_v1'
                or meta['features'] != [{'name': x, 'dtype': 'float64'} for x in FEATURES]
                or meta['horizons'] != list(range(1, 25)) or meta['feature_version'] != FEATURE_VERSION
                or meta['inference_policy'] != POLICY):
            raise ValueError('Unsupported/corrupt curve metadata')
        expected = {f'{c}_h{h:02d}' for c in COMPONENTS for h in range(1, 25)}
        if set(meta['entries']) != expected:
            raise ValueError('Expected all 72 component/horizon candidates')
        self.models, self.errors = {}, {}
        for key, entry in meta['entries'].items():
            try:
                if (key != f"{entry['component']}_h{entry['horizon']:02d}"
                        or entry['selected_method'] not in (*BASELINES, METHOD)
                        or set(entry['baseline_ranking']) != set(BASELINES)
                        or len(entry['baseline_ranking']) != 3):
                    raise ValueError('Invalid curve selection')
                parse_origin(entry['available_after_target'])
                self.models[key] = load_candidate(self.directory, entry)
            except (ValueError, KeyError, OSError) as exc:
                self.errors[key] = str(exc)
        if strict and self.errors:
            raise ValueError(f'Curve preflight failed: {self.errors}')

    def forecast_rows(self, context, origins, baseline_only=False):
        """Efficient causal replay; one row per (origin,horizon), no actuals."""
        origins = pd.DatetimeIndex(origins)
        if origins.has_duplicates or not origins.isin(context.index[context.present]).all():
            raise ValueError('Origins must be unique observed hours')
        output = []
        for h in range(1, 25):
            frame = pd.DataFrame({'origin': origins, 'horizon': h,
                                  'target_timestamp': origins+h*HOUR})
            for c in COMPONENTS:
                key = f'{c}_h{h:02d}'
                entry = self.manifest['entries'][key]
                known = (origins > pd.Timestamp(entry['available_after_target'])) & (key not in self.errors)
                requested = entry['baseline_ranking'][0] if baseline_only else entry['selected_method']
                features = context.features(c, h, origins)
                checks = context.quality(c, origins, None if requested == METHOD else baseline_offset(requested, h))
                checks['selection_not_available_at_origin'] = ~known
                if key in self.errors:
                    checks['artifact_unavailable'] = True
                values = np.full(len(origins), np.nan)
                methods = np.full(len(origins), None, dtype=object)
                status = np.full(len(origins), 'blocked', dtype=object)
                reasons = _reason_lists(checks)
                unavailable = context.unavailable(c, origins, None if requested == METHOD else baseline_offset(requested, h))
                if requested == METHOD:
                    good = ~checks.any(axis=1).to_numpy() & np.isfinite(features.to_numpy()).all(axis=1)
                    if good.any():
                        pred = predict_candidate(self.models[key], features.loc[good], entry)
                        valid = np.isfinite(pred) & ((pred > 0) if c == 'demand' else (pred >= 0))
                        positions = np.flatnonzero(good)
                        values[positions[valid]] = pred[valid]
                        for i in positions[~valid]:
                            reasons[i].append('invalid_model_output')
                else:
                    good = ~checks.any(axis=1).to_numpy()
                    values[good] = context.baseline(c, requested, h, origins)[good]
                initial = np.isfinite(values)
                methods[initial], status[initial] = requested, 'ok'
                for rank in range(3):
                    for method in BASELINES:
                        choice = np.where(known, entry['baseline_ranking'][rank], BASELINES[rank]) == method
                        pending = ~np.isfinite(values) & choice
                        if not pending.any():
                            continue
                        flags = context.quality(c, origins, baseline_offset(method, h))
                        usable = pending & ~flags.any(axis=1).to_numpy()
                        candidate = context.baseline(c, method, h, origins)
                        values[usable] = candidate[usable]
                        methods[usable], status[usable] = method, 'fallback'
                        notes = context.unavailable(c, origins, baseline_offset(method, h))
                        unavailable.loc[usable] |= notes.loc[usable]
                        for i in np.flatnonzero(pending & ~usable):
                            reasons[i] += [f'{method}:{name}' for name, active in flags.iloc[i].items() if active]
                frame[f'{c}_mw'] = values
                frame[f'{c}_method'] = methods
                frame[f'{c}_status'] = status
                frame[f'{c}_reasons'] = [json.dumps(sorted(set(r))) for r in reasons]
                frame[f'{c}_unavailable_checks'] = [json.dumps(r) for r in _reason_lists(unavailable)]
            frame['net_demand_mw'] = frame.demand_mw-frame.wind_mw-frame.solar_mw
            frame['renewable_mw'] = frame.wind_mw+frame.solar_mw
            frame['target_quality'] = 'unknown'
            output.append(frame)
        return pd.concat(output, ignore_index=True).sort_values(['origin', 'horizon']).reset_index(drop=True)

    def forecast(self, history, origin):
        origin = parse_origin(origin)
        context = CurveHistory(history, origin)
        rows = self.forecast_rows(context, [origin])
        points = []
        for _, row in rows.iterrows():
            components = {}
            for c in COMPONENTS:
                entry = self.manifest['entries'][f'{c}_h{int(row.horizon):02d}']
                components[c] = {'forecast_mw': None if pd.isna(row[f'{c}_mw']) else float(row[f'{c}_mw']),
                    'method': row[f'{c}_method'], 'status': row[f'{c}_status'],
                    'reasons': json.loads(row[f'{c}_reasons']),
                    'unavailable_quality_checks': json.loads(row[f'{c}_unavailable_checks']),
                    'selected_method': entry['selected_method'], 'model_sha256': entry['model_sha256']}
            points.append({'horizon': int(row.horizon), 'target_timestamp': str(row.target_timestamp),
                           'components': components,
                           'net_demand_mw': None if pd.isna(row.net_demand_mw) else float(row.net_demand_mw),
                           'renewable_mw': None if pd.isna(row.renewable_mw) else float(row.renewable_mw),
                           'target_quality': 'unknown'})
        statuses = rows[[f'{c}_status' for c in COMPONENTS]]
        blocked = statuses.eq('blocked').any(axis=1)
        return {'schema_version': 1, 'model_id': 'grid_curve_v1', 'origin': str(origin),
                'horizons': list(range(1, 25)), 'timestamp_convention': 'timezone-naive source clock; no conversion',
                'origin_observation_assumed_available': True,
                'run_context': 'retrospective_simulation; not evidence of an actually issued forecast',
                'inference_quality_policy': POLICY, 'target_quality': 'unknown',
                'status': 'blocked' if blocked.all() else ('degraded' if not statuses.eq('ok').all().all() else 'ok'),
                'complete_curve': not bool(blocked.any()), 'delivered_target_hours': int((~blocked).sum()),
                'manifest_sha256': digest(self.directory/'metadata.json'), 'points': points}
