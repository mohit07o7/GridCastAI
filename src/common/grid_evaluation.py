"""Pure alignment, target-audit and paired scoring helpers for grid evaluation."""
import json

import numpy as np
import pandas as pd

KEYS = ['origin', 'target_timestamp']
COMPONENTS = ['demand', 'wind', 'solar']
EVALUATION_POLICY = 'grid_evaluation_target_v1'


def validate_keys(frame, label):
    frame = frame.copy()
    for key in KEYS:
        times = pd.to_datetime(frame[key], errors='raise')
        if times.isna().any() or times.dt.tz is not None or times.ne(times.dt.floor('h')).any():
            raise ValueError(f'{label}: invalid hourly source-clock {key}')
        frame[key] = times
    if frame.duplicated(KEYS).any():
        raise ValueError(f'{label}: duplicate origin/target keys')
    if frame.target_timestamp.sub(frame.origin).ne(pd.Timedelta(hours=24)).any():
        raise ValueError(f'{label}: target must be exactly origin + 24h')
    return frame.sort_values(KEYS).reset_index(drop=True)


def validate_component(frame, source, component):
    frame = validate_keys(frame, component)
    for prefix, times in [('actual', frame.target_timestamp), ('baseline', frame.origin)]:
        values = frame[f'{prefix}_{component}'].to_numpy(dtype=float)
        expected = source[f'{component}_mw'].reindex(times).to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(expected).all():
            raise ValueError(f'{component}: nonfinite/missing {prefix} alignment')
        if not np.allclose(values, expected, rtol=0, atol=1e-6):
            raise ValueError(f'{component}: incorrect {prefix} timestamp alignment')
    if not np.isfinite(frame[f'predicted_{component}']).all():
        raise ValueError(f'{component}: nonfinite saved prediction')
    return frame


def attach_observations(frame, source):
    """Evaluation-only attachment, called AFTER prediction generation."""
    result = validate_keys(frame, 'observations')
    for component in COMPONENTS:
        for prefix, times in [('actual', result.target_timestamp), ('baseline', result.origin)]:
            result[f'{prefix}_{component}'] = source[f'{component}_mw'].reindex(times).to_numpy()
    return result


def combined_quantities(frame):
    result = frame.copy()
    for prefix in ['actual', 'predicted', 'baseline']:
        demand = result[f'{prefix}_demand']
        renewable = result[f'{prefix}_wind'] + result[f'{prefix}_solar']
        result[f'{prefix}_renewable_mw'] = renewable
        result[f'{prefix}_net_demand_mw'] = demand - renewable
        result[f'{prefix}_renewable_share_pct'] = 100 * renewable / demand.where(demand.gt(0))
        result[f'{prefix}_share_undefined'] = ~np.isfinite(result[f'{prefix}_renewable_share_pct'])
    for prefix in ['predicted', 'baseline']:
        for component in COMPONENTS:
            result[f'{prefix}_{component}_error_mw'] = result[f'{prefix}_{component}'] - result[f'actual_{component}']
        error = result[f'{prefix}_net_demand_mw'] - result.actual_net_demand_mw
        expected = result[f'{prefix}_demand_error_mw'] - result[f'{prefix}_wind_error_mw'] - result[f'{prefix}_solar_error_mw']
        finite = np.isfinite(error) & np.isfinite(expected)
        if not np.allclose(error[finite], expected[finite], rtol=0, atol=1e-8):
            raise ValueError('Net-demand error identity failed')
        result[f'{prefix}_net_error_mw'] = error
        result[f'{prefix}_net_absolute_error_mw'] = error.abs()
        result[f'{prefix}_renewable_error_mw'] = result[f'{prefix}_renewable_mw'] - result.actual_renewable_mw
        result[f'{prefix}_renewable_share_error_pp'] = result[f'{prefix}_renewable_share_pct'] - result.actual_renewable_share_pct
    return result


def _frozen(values, index):
    # Same >=6-hour definition as the historical audit; gaps break episodes.
    contiguous = index.to_series().diff().eq(pd.Timedelta(hours=1))
    groups = (~(values.eq(values.shift()) & contiguous)).cumsum()
    return groups.map(groups.value_counts()).ge(6) & np.isfinite(values)


def retrospective_target_flags(source):
    """Full-data retrospective scoring flags. NEVER passed to prediction."""
    values = source[[f'{c}_mw' for c in COMPONENTS]]
    flags = pd.DataFrame(index=source.index)
    flags['target_invalid_components'] = ~np.isfinite(values).all(axis=1) | values.lt(0).any(axis=1)
    flags['target_nonpositive_demand'] = source.demand_mw.le(0)
    flags['target_all_zero'] = values.eq(0).all(axis=1)
    flags['target_april_anomaly'] = source.index == pd.Timestamp('2025-04-04 16:00:00')
    flags['target_frozen_demand'] = _frozen(source.demand_mw, source.index)
    flags['target_frozen_wind'] = _frozen(source.wind_mw, source.index)
    flags['target_positive_solar_flatline'] = _frozen(source.solar_mw, source.index) & source.solar_mw.gt(500)
    if 'solar_wind_mw' in source:
        flags['target_renewable_inconsistency'] = (source.solar_wind_mw-source.wind_mw-source.solar_mw).abs().gt(1)
    else:
        flags['target_renewable_inconsistency'] = False
    return flags


def attach_target_flags(frame, flags):
    result = frame.copy()
    aligned = flags.reindex(pd.DatetimeIndex(result.target_timestamp))
    for column in flags:
        result[column] = aligned[column].fillna(False).to_numpy(dtype=bool)
    result['target_missing_observation'] = ~result.target_timestamp.isin(flags.index)
    columns = list(flags) + ['target_missing_observation']
    result['trusted_target'] = ~result[columns].any(axis=1)
    result['target_exclusion_reasons'] = result[columns].apply(
        lambda row: json.dumps([name for name, active in row.items() if active]), axis=1)
    return result


def assign_replay_cohorts(frame):
    result = frame.copy()
    statuses = result[[f'{c}_status' for c in COMPONENTS]]
    if not statuses.isin(['ok', 'fallback', 'blocked']).all().all():
        raise ValueError('Unrecognized component status')
    available = np.isfinite(result[[f'predicted_{c}' for c in COMPONENTS]]).all(axis=1)
    for component in COMPONENTS:
        finite = np.isfinite(result[f'predicted_{component}'])
        blocked = result[f'{component}_status'].eq('blocked')
        if (blocked & finite).any() or (~blocked & ~finite).any():
            raise ValueError('Blocked forecasts must be missing; delivered forecasts must be finite')
        if (result[f'predicted_{component}'].lt(0) & ~blocked).any():
            raise ValueError('Delivered component forecasts must be nonnegative')
    result['baseline_available'] = np.isfinite(result[[f'baseline_{c}' for c in COMPONENTS]]).all(axis=1)
    result['baseline_available'] &= result[[f'baseline_{c}' for c in COMPONENTS]].ge(0).all(axis=1)
    result['selected_methods_available'] = statuses.eq('ok').all(axis=1) & available
    result['delivered_forecast_available'] = ~statuses.eq('blocked').any(axis=1) & available
    result['any_fallback'] = statuses.eq('fallback').any(axis=1)
    for cohort, availability in [('selected_only', result.selected_methods_available),
                                 ('delivered', result.delivered_forecast_available)]:
        result[f'score_{cohort}'] = result.trusted_target & result.baseline_available & availability
    return result


def _score(actual, predicted):
    if len(actual) == 0:
        return {'mae_mw': None, 'rmse_mw': None, 'wape_pct': None, 'mean_signed_error_mw': None}
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {'mae_mw': float(np.abs(error).mean()), 'rmse_mw': float(np.sqrt(np.mean(error**2))),
            'wape_pct': float(100*np.abs(error).sum()/denominator) if denominator > 0 else None,
            'mean_signed_error_mw': float(error.mean())}


def paired_day_bootstrap(frame, repeats=1000, seed=42):
    if len(frame) == 0:
        return {'mae_improvement_mw': None, 'ci95_low_mw': None, 'ci95_high_mw': None, 'days': 0, 'repeats': repeats, 'seed': seed}
    paired = pd.DataFrame({'day': frame.target_timestamp.dt.date,
                           'delta': frame.baseline_net_absolute_error_mw-frame.predicted_net_absolute_error_mw})
    daily = paired.groupby('day').delta.agg(['sum','count'])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(daily), size=(repeats, len(daily)))
    distribution = daily['sum'].to_numpy()[draws].sum(axis=1)/daily['count'].to_numpy()[draws].sum(axis=1)
    return {'mae_improvement_mw': float(paired.delta.mean()), 'ci95_low_mw': float(np.percentile(distribution,2.5)),
            'ci95_high_mw': float(np.percentile(distribution,97.5)), 'days': len(daily), 'repeats': repeats, 'seed': seed}


def paired_summary(frame, mask, cohort, nominal_rows, trusted_rows, bootstrap=True):
    selected = frame.loc[mask].copy()
    required = [f'{prefix}_{kind}' for prefix in ['actual','predicted','baseline']
                for kind in ['net_demand_mw','renewable_mw']]
    if not np.isfinite(selected[required]).all().all():
        raise ValueError('Cannot score nonfinite paired values; explicit eligibility is required')
    result = {'cohort':cohort, 'rows':len(selected), 'nominal_rows':nominal_rows, 'trusted_target_rows':trusted_rows,
              'coverage_nominal_pct':100*len(selected)/nominal_rows if nominal_rows else None,
              'coverage_trusted_pct':100*len(selected)/trusted_rows if trusted_rows else None,
              'target_start':str(selected.target_timestamp.min()) if len(selected) else None,
              'target_end':str(selected.target_timestamp.max()) if len(selected) else None}
    shares = [f'{prefix}_renewable_share_pct' for prefix in ['actual','predicted','baseline']]
    share_mask = np.isfinite(selected[shares]).all(axis=1)
    result['paired_share_rows'] = int(share_mask.sum())
    for prefix in ['predicted','baseline']:
        for kind in ['net','renewable']:
            column = 'net_demand_mw' if kind=='net' else 'renewable_mw'
            scores = _score(selected[f'actual_{column}'].to_numpy(), selected[f'{prefix}_{column}'].to_numpy())
            result.update({f'{prefix}_{kind}_{k}':v for k,v in scores.items()})
        result[f'{prefix}_renewable_share_mae_pp'] = float(selected.loc[share_mask,f'{prefix}_renewable_share_error_pp'].abs().mean()) if share_mask.any() else None
        result[f'{prefix}_undefined_share_rows'] = int(selected[f'{prefix}_share_undefined'].sum())
    improvement = result['baseline_net_mae_mw'] - result['predicted_net_mae_mw'] if len(selected) else None
    result['mae_improvement_mw'] = improvement
    result['mae_improvement_pct'] = 100*improvement/result['baseline_net_mae_mw'] if len(selected) and result['baseline_net_mae_mw'] > 0 else None
    if bootstrap:
        result['bootstrap'] = paired_day_bootstrap(selected)
    return result
