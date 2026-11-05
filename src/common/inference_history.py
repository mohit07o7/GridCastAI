"""Origin-only inputs. No future labels or retrospective training builders."""
import numpy as np
import pandas as pd

DEMAND_COLUMNS = [
    'demand_lag_1', 'demand_lag_24', 'demand_lag_48', 'demand_lag_168',
    'demand_mean_24h', 'demand_mean_168h', 'wind_lag_1', 'wind_lag_24',
    'solar_lag_1', 'solar_lag_24', 'hour', 'day_of_week', 'month',
]
WIND_COLUMNS = [
    'prior_day', 'origin_lag_1', 'origin_lag_24', 'origin_lag_48',
    'origin_lag_168', 'target_365d_ago', 'recent_mean_24', 'recent_std_24',
    'recent_mean_168', 'recent_std_168', 'hour_sin', 'hour_cos', 'day_sin',
    'day_cos', 'target_year', 'target_month',
]
POLICY_VERSION = 'inference_quality_v1'
MEASUREMENTS = ['demand_mw', 'wind_mw', 'solar_mw']
HOUR = pd.Timedelta(hours=1)


class HistoryError(ValueError):
    """Invalid request or unambiguous history cannot be established."""


def parse_origin(value):
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise HistoryError('Origin must be an exact timezone-naive hourly timestamp') from exc
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.floor('h'):
        raise HistoryError('Origin must be an exact timezone-naive hourly timestamp')
    return stamp


def prepare_history(data, origin):
    """Normalize timestamps, cut future rows, then inspect past measurements."""
    origin = parse_origin(origin)
    if not isinstance(data, pd.DataFrame) or data.columns.has_duplicates:
        raise HistoryError('History must have unique column names')
    stamp_cols = [c for c in ['Timestamp', 'timestamp'] if c in data]
    if not stamp_cols or any(c not in data for c in MEASUREMENTS):
        raise HistoryError('Require Timestamp or timestamp, demand_mw, wind_mw and solar_mw')
    parsed = []
    for column in stamp_cols:
        try:
            times = pd.to_datetime(data[column], format='mixed', errors='raise')
            if times.isna().any() or times.dt.tz is not None:
                raise ValueError('Missing/timezone-aware timestamps')
        except (ValueError, TypeError, AttributeError) as exc:
            raise HistoryError('History timestamps must be valid and timezone-naive') from exc
        parsed.append(times)
    if len(parsed) == 2 and not parsed[0].equals(parsed[1]):
        raise HistoryError('Conflicting Timestamp and timestamp columns')
    # Measurement validation and duplicate checks happen only after this cutoff.
    past = data.loc[parsed[0] <= origin].copy()
    past.index = pd.DatetimeIndex(parsed[0].loc[parsed[0] <= origin], name='origin')
    past = past.drop(columns=stamp_cols).sort_index()
    if past.index.has_duplicates:
        raise HistoryError('Ambiguous duplicate observations at or before origin')
    if origin not in past.index:
        raise HistoryError('No observation exists at the requested origin')
    if not past.index.equals(past.index.floor('h')):
        raise HistoryError('History observations must fall on exact hours')
    for column in MEASUREMENTS + ['solar_wind_mw']:
        if column in past:
            invalid_type = past[column].map(lambda v: isinstance(v, (bool, complex, np.bool_, np.complexfloating)))
            past[column] = pd.to_numeric(past[column], errors='coerce').astype(float).mask(invalid_type)
    return past


def recent_index(origin):
    return pd.date_range(origin - 168 * HOUR, origin, freq='h')


def _observed_window(past, origin):
    # Pure feature builders also enforce a cutoff for callers using indexed data.
    history = past.loc[:origin]
    index = recent_index(origin)
    if not index.isin(history.index).all():
        raise ValueError('missing_recent_hourly_history')
    return history, history.loc[index]


def demand_features(past, origin):
    origin = parse_origin(origin)
    history, window = _observed_window(past, origin)
    values = {}
    for lag in [1, 24, 48, 168]:
        values[f'demand_lag_{lag}'] = window.at[origin - lag * HOUR, 'demand_mw']
    for hours in [24, 168]:
        values[f'demand_mean_{hours}h'] = window.loc[origin - hours * HOUR:origin - HOUR, 'demand_mw'].mean(skipna=False)
    for component in ['wind', 'solar']:
        for lag in [1, 24]:
            values[f'{component}_lag_{lag}'] = window.at[origin - lag * HOUR, f'{component}_mw']
    values.update(hour=origin.hour, day_of_week=origin.dayofweek, month=origin.month)
    return pd.DataFrame([values], columns=DEMAND_COLUMNS)


def wind_features(past, origin):
    origin = parse_origin(origin)
    history, window = _observed_window(past, origin)
    annual = origin - 8736 * HOUR
    if annual not in history.index:
        raise ValueError('missing_wind_annual_reference')
    value = window.wind_mw
    target = origin + 24 * HOUR
    values = {'prior_day': value.at[origin]}
    for lag in [1, 24, 48, 168]:
        values[f'origin_lag_{lag}'] = value.at[origin - lag * HOUR]
    values['target_365d_ago'] = history.at[annual, 'wind_mw']
    for hours in [24, 168]:
        latest = value.iloc[-hours:]
        values[f'recent_mean_{hours}'] = latest.mean(skipna=False)
        values[f'recent_std_{hours}'] = latest.std(ddof=1, skipna=False)
    values.update(hour_sin=np.sin(2 * np.pi * target.hour / 24),
                  hour_cos=np.cos(2 * np.pi * target.hour / 24),
                  day_sin=np.sin(2 * np.pi * target.dayofyear / 365.25),
                  day_cos=np.cos(2 * np.pi * target.dayofyear / 365.25),
                  target_year=target.year, target_month=target.month)
    return pd.DataFrame([values], columns=WIND_COLUMNS)


def quality_flags(past):
    """All evidence is in this origin-truncated frame; gaps break frozen runs."""
    contiguous = past.index.to_series().diff().eq(HOUR)
    finite = np.isfinite(past[MEASUREMENTS])
    flags = {}
    for component in ['demand', 'wind', 'solar']:
        value = past[f'{component}_mw']
        valid = np.isfinite(value) & value.ge(0)
        same = value.eq(value.shift()) & contiguous & valid
        groups = (~same).cumsum()
        frozen = groups.map(groups.value_counts()).ge(6) & valid
        if component == 'solar':
            frozen &= value.gt(500)
        flags[component] = pd.DataFrame({'invalid_measurement': ~valid, 'frozen_reading': frozen})
    zero = past[MEASUREMENTS].eq(0).all(axis=1) & finite.all(axis=1)
    previous = past.demand_mw.shift()
    jumps = contiguous & previous.gt(0) & np.isfinite(previous) & np.isfinite(past.demand_mw)
    jumps &= ((past.demand_mw - previous).abs() / previous).gt(.40)
    flags['demand']['sharp_demand_change'] = jumps
    if 'solar_wind_mw' in past:
        balance_available = np.isfinite(past[['solar_wind_mw', 'wind_mw', 'solar_mw']]).all(axis=1)
        imbalance = balance_available & (past.solar_wind_mw - past.wind_mw - past.solar_mw).abs().gt(1)
    else:
        balance_available = pd.Series(False, index=past.index)
        imbalance = balance_available.copy()
    for component in flags:
        flags[component]['all_zero_record'] = zero
        if component in ['wind', 'solar']:
            flags[component]['renewable_component_inconsistency'] = imbalance
    return flags, finite.all(axis=1), balance_available


def check_quality(flags, timestamps, component):
    index = pd.DatetimeIndex(timestamps)
    present = index.intersection(flags[component].index)
    reasons = []
    if len(present) != len(index):
        reasons.append(f'{component}:missing_required_history')
    if len(present):
        reasons += [f'{component}:{column}' for column, active in flags[component].loc[present].any().items() if active]
    return reasons
