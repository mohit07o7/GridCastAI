"""Shared origin-relative, 24-hour-ahead demand features."""
import pandas as pd
DEMAND_FEATURES = ['demand_lag_1', 'demand_lag_24', 'demand_lag_48', 'demand_lag_168', 'demand_mean_24h', 'demand_mean_168h', 'hour', 'day_of_week', 'month']
RENEWABLE_FEATURES = ['wind_lag_1', 'wind_lag_24', 'solar_lag_1', 'solar_lag_24']
TREND_FEATURES = ['demand_trend_6h', 'demand_trend_24h']

def build_features(data):
    df = data.sort_values('Timestamp').reset_index(drop=True).copy()
    if df.Timestamp.isna().any() or df.Timestamp.duplicated().any() or df.Timestamp.diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError('Expected unique, continuous hourly timestamps')
    if df[['demand_mw', 'wind_mw', 'solar_mw']].isna().any().any():
        raise ValueError('Missing source measurements')
    df['Target'] = df.demand_mw.shift(-24)
    df['Target Timestamp'] = df.Timestamp + pd.Timedelta(hours=24)
    for lag in [1, 24, 48, 168]:
        df[f'demand_lag_{lag}'] = df.demand_mw.shift(lag)
    for hours in [24, 168]:
        df[f'demand_mean_{hours}h'] = df.demand_mw.shift(1).rolling(hours).mean()
    for lag in [1, 24]:
        for col in ['wind', 'solar']:
            df[f'{col}_lag_{lag}'] = df[f'{col}_mw'].shift(lag)
    for hours in [6, 24]:
        df[f'demand_trend_{hours}h'] = df.demand_mw.shift(1) - df.demand_mw.shift(hours + 1)
    df['hour'] = df.Timestamp.dt.hour
    df['day_of_week'] = df.Timestamp.dt.dayofweek
    df['month'] = df.Timestamp.dt.month
    # Demand quality v1: preserve source values and flag eligibility separately.
    zero = df[['demand_mw', 'wind_mw', 'solar_mw']].eq(0).all(axis=1)
    groups = df.demand_mw.ne(df.demand_mw.shift()).cumsum()
    frozen = df.groupby(groups).demand_mw.transform('size').ge(6)
    april = df.Timestamp.eq(pd.Timestamp('2025-04-04 16:00:00'))
    df['demand_zero_flag'] = zero
    df['demand_frozen_flag'] = frozen
    df['demand_april_flag'] = april
    bad = zero | frozen | april
    df['quality_target_bad'] = bad.shift(-24, fill_value=False)
    df['quality_input_bad'] = bad.astype(int).rolling(169, min_periods=1).max().astype(bool)
    df['quality_eligible'] = ~(df.quality_target_bad | df.quality_input_bad)
    return df
