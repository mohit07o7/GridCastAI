"""Causal features and observed quality for direct horizons 1--24.

Batch calculations are algebraically identical to cutting the input at each
origin. In particular, a frozen episode becomes observable at its sixth hour;
its eventual length never decides whether an earlier origin is eligible.
"""
import numpy as np
import pandas as pd

from common.inference_history import HOUR, HistoryError, parse_origin, prepare_history

COMPONENTS = ('demand', 'wind', 'solar')
POLICY = 'curve_inference_quality_v1'
FEATURE_VERSION = 'direct_curve_features_v1'
BASELINES = ('previous_day', 'last_observation', 'previous_week')
VALIDATION_START = pd.Timestamp('2024-05-03 02:00:00')
TEST_START = pd.Timestamp('2024-11-28 01:00:00')
FEATURES = ([f'lag_{i}' for i in [0, 1, 3, 6, 24, 48, 168]]
            + [f'recent_{stat}_{w}' for w in [24, 168] for stat in ['mean', 'std']]
            + ['target_previous_day', 'target_previous_week', 'target_hour_sin',
               'target_hour_cos', 'target_dow_sin', 'target_dow_cos',
               'target_month_sin', 'target_month_cos', 'target_year'])


def baseline_offset(method, horizon):
    if method not in BASELINES or type(horizon) is not int or not 1 <= horizon <= 24:
        raise ValueError('Unsupported baseline or horizon')
    return {'previous_day': 24-horizon, 'last_observation': 0,
            'previous_week': 168-horizon}[method]


class CurveHistory:
    """Normalized observations; features contain no labels or target flags."""
    def __init__(self, data, origin=None):
        if origin is None:
            column = 'timestamp' if 'timestamp' in data else 'Timestamp'
            origin = pd.to_datetime(data[column], format='mixed', errors='raise').max()
        past = prepare_history(data, origin)
        self.index = pd.date_range(past.index.min(), past.index.max(), freq='h', name='origin')
        self.data = past.reindex(self.index)
        self.present = pd.Series(self.index.isin(past.index), index=self.index)
        self.positions = pd.Series(np.arange(len(self.index)), index=self.index)
        vals = self.data[[f'{c}_mw' for c in COMPONENTS]]
        self.zero_available = np.isfinite(vals).all(axis=1)
        all_zero = vals.eq(0).all(axis=1) & self.zero_available
        if 'solar_wind_mw' in self.data:
            cols = self.data[['solar_wind_mw', 'wind_mw', 'solar_mw']]
            self.balance_available = np.isfinite(cols).all(axis=1)
            imbalance = self.balance_available & (cols.solar_wind_mw-cols.wind_mw-cols.solar_mw).abs().gt(1)
        else:
            self.balance_available = pd.Series(False, index=self.index)
            imbalance = self.balance_available.copy()
        self.point_flags, self.detection, self.window_flags, self.base_features = {}, {}, {}, {}
        self.context = {}
        for c in COMPONENTS:
            value = self.data[f'{c}_mw']
            valid = np.isfinite(value) & (value.gt(0) if c == 'demand' else value.ge(0))
            same = valid & valid.shift(fill_value=False) & value.eq(value.shift())
            groups = (~same).cumsum()
            length = groups.map(groups.value_counts())
            start = self.positions.groupby(groups).transform('min')
            detectable = valid & length.ge(6)
            if c == 'solar':
                detectable &= value.gt(500)
            # Full-series runs only encode their earliest POSSIBLE detection time.
            # Every query compares that time with its origin; future evidence is inert.
            detection = (start+5).where(detectable, np.inf)
            self.detection[c] = detection
            flags = pd.DataFrame({'invalid_measurement': ~valid, 'all_zero_record': all_zero})
            if c == 'demand':
                previous = value.shift()
                flags['sharp_demand_change'] = previous.gt(0) & np.isfinite(previous) & ((value-previous).abs()/previous).gt(.40)
            else:
                flags['renewable_component_inconsistency'] = imbalance
            self.point_flags[c] = flags
            window_flags = flags.copy()
            window_flags['frozen_reading'] = self.positions.ge(detection)
            self.window_flags[c] = window_flags.astype(int).rolling(169, min_periods=1).max().astype(bool)
            self.window_flags[c]['missing_required_history'] = self.present.astype(int).rolling(169, min_periods=169).sum().ne(169)
            base = pd.DataFrame({f'lag_{lag}': value.shift(lag) for lag in [0, 1, 3, 6, 24, 48, 168]})
            for w in [24, 168]:
                base[f'recent_mean_{w}'] = value.rolling(w, min_periods=w).mean()
                base[f'recent_std_{w}'] = value.rolling(w, min_periods=w).std(ddof=1)
            self.base_features[c] = base
            self.context[c] = np.isfinite(value).astype(int).rolling(6, min_periods=6).sum().eq(6)

    def features(self, component, horizon, origins=None):
        if component not in COMPONENTS or type(horizon) is not int or not 1 <= horizon <= 24:
            raise ValueError('Expected component and integer horizon 1--24')
        origins = self.index if origins is None else pd.DatetimeIndex(origins)
        result = self.base_features[component].reindex(origins).copy()
        value = self.data[f'{component}_mw']
        for name, lag in [('day', 24), ('week', 168)]:
            result[f'target_previous_{name}'] = value.reindex(origins-(lag-horizon)*HOUR).to_numpy()
        targets = origins + horizon*HOUR
        for name, values, period in [('hour', targets.hour, 24), ('dow', targets.dayofweek, 7),
                                     ('month', targets.month-1, 12)]:
            result[f'target_{name}_sin'] = np.sin(2*np.pi*values/period)
            result[f'target_{name}_cos'] = np.cos(2*np.pi*values/period)
        result['target_year'] = targets.year.astype(float)
        return result.loc[:, FEATURES].astype(float)

    def baseline(self, component, method, horizon, origins):
        origins = pd.DatetimeIndex(origins)
        times = origins-baseline_offset(method, horizon)*HOUR
        return self.data[f'{component}_mw'].reindex(times).to_numpy(dtype=float)

    def quality(self, component, origins, offset=None):
        """Return origin-indexed reason masks; offset=None checks model history."""
        origins = pd.DatetimeIndex(origins)
        if offset is None:
            return self.window_flags[component].reindex(origins, fill_value=True).copy()
        refs = origins-offset*HOUR
        flags = self.point_flags[component].reindex(refs, fill_value=True).copy()
        flags.index = origins
        detect = self.detection[component].reindex(refs).to_numpy()
        positions = self.positions.reindex(origins).to_numpy()
        flags['frozen_reading'] = positions >= detect
        flags['missing_required_history'] = ~refs.isin(self.index[self.present])
        return flags

    def unavailable_checks(self, component, origin, offset=None):
        origin = parse_origin(origin)
        times = pd.date_range(origin-168*HOUR, origin, freq='h') if offset is None else pd.DatetimeIndex([origin-offset*HOUR])
        notes = []
        if not self.zero_available.reindex(times, fill_value=False).all():
            notes.append('all_zero_check_incomplete')
        if component != 'demand' and not self.balance_available.reindex(times, fill_value=False).all():
            notes.append('renewable_component_balance_check_incomplete')
        value = self.data[f'{component}_mw']
        context = self.context[component]
        if not context.reindex(times, fill_value=False).all():
            notes.append('six_hour_flatline_context_incomplete')
        if component == 'demand':
            prev = self.present & np.isfinite(value)
            if not prev.reindex(times-HOUR, fill_value=False).all():
                notes.append('sharp_change_previous_hour_missing')
        return notes

    def unavailable(self, component, origins, offset=None):
        origins = pd.DatetimeIndex(origins)
        checks = {'all_zero_check_incomplete': ~self.zero_available,
                  'six_hour_flatline_context_incomplete': ~self.context[component]}
        if component != 'demand':
            checks['renewable_component_balance_check_incomplete'] = ~self.balance_available
        else:
            checks['sharp_change_previous_hour_missing'] = ~(self.present & np.isfinite(self.data.demand_mw)).shift(fill_value=False)
        result = pd.DataFrame(index=origins)
        for name, flags in checks.items():
            if offset is None:
                values = flags.astype(int).rolling(169, min_periods=1).max().astype(bool)
                result[name] = values.reindex(origins, fill_value=True).to_numpy()
            else:
                result[name] = flags.reindex(origins-offset*HOUR, fill_value=True).to_numpy()
        return result


def split_masks(origins):
    origins = pd.DatetimeIndex(origins)
    masks = {'train': (origins+24*HOUR < VALIDATION_START),
             'validation': (origins >= VALIDATION_START) & (origins+24*HOUR < TEST_START),
             'test': origins >= TEST_START}
    if masks['train'].any() and masks['validation'].any():
        assert (origins[masks['train']]+24*HOUR).max() < origins[masks['validation']].min()
    if masks['validation'].any() and masks['test'].any():
        assert (origins[masks['validation']]+24*HOUR).max() < origins[masks['test']].min()
    return masks
