import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.grid_prediction import GridForecaster, ROOT, load_wind_model
from common.inference_history import (
    prepare_history, quality_flags, demand_features, wind_features, HistoryError,
)


class GridPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / 'data/processed/full_hourly_stacked.csv')
        cls.engine = GridForecaster(ROOT)
        cls.origin = pd.Timestamp('2025-06-24 23:00:00')

    def setUp(self):
        self.patches = [patch.object(xgboost.XGBRegressor, 'fit', side_effect=AssertionError('No training')),
                        patch.object(xgboost, 'train', side_effect=AssertionError('No training')),
                        patch.object(xgboost.Booster, 'update', side_effect=AssertionError('No training'))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def synthetic(self, count=9000):
        n = np.arange(count)
        return pd.DataFrame({'Timestamp': pd.date_range(end=self.origin, periods=count, freq='h'),
                             'demand_mw': 200000 + n * 2., 'wind_mw': 10000 + n * .1,
                             'solar_mw': 1000 + n * .01})

    def test_normal_end_at_origin_and_solar_selected(self):
        report = self.engine.forecast(self.raw, self.origin)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['target_timestamp'], '2025-06-25 23:00:00')
        self.assertEqual(report['target_quality'], 'unknown')
        solar = report['components']['solar']
        self.assertEqual(solar['status'], 'ok')
        self.assertEqual(solar['forecast_mw'], self.raw.solar_mw.iloc[-1])
        self.assertTrue(all(v['target_timestamp'] == report['target_timestamp'] for v in report['components'].values()))

    def test_future_and_sorting_invariance(self):
        origin = pd.Timestamp('2025-05-05 15:00:00')
        data = self.raw.copy()
        expected = self.engine.forecast(data, origin)
        future = pd.to_datetime(data.timestamp) > origin
        data.loc[future, ['demand_mw', 'wind_mw', 'solar_mw', 'solar_wind_mw']] = np.nan
        data = pd.concat([data, data.loc[future].iloc[:2]], ignore_index=True)
        self.assertEqual(expected, self.engine.forecast(data.sample(frac=1, random_state=42), origin))
        truncated = self.raw.loc[pd.to_datetime(self.raw.timestamp) <= origin].rename(columns={'timestamp':'Timestamp'})
        truncated = truncated[['Timestamp','demand_mw','wind_mw','solar_mw','solar_wind_mw']]
        self.assertEqual(expected, self.engine.forecast(truncated, origin))

    def test_required_recent_and_annual_history(self):
        short = self.synthetic(169)
        report = self.engine.forecast(short, self.origin)['components']
        self.assertEqual(report['demand']['status'], 'ok')
        self.assertEqual(report['wind']['status'], 'fallback')
        missing = self.synthetic().drop(index=[100, 8990])
        report = self.engine.forecast(missing, self.origin)['components']
        self.assertEqual(report['demand']['status'], 'fallback')
        self.assertEqual(report['wind']['status'], 'fallback')
        self.assertEqual(report['solar']['status'], 'ok')
        tiny = self.synthetic(1)
        self.assertEqual(self.engine.forecast(tiny,self.origin)['components']['demand']['status'],'fallback')

    def test_invalid_channel_is_isolated(self):
        data = self.synthetic()
        data.loc[data.index[-1], 'wind_mw'] = np.inf
        result = self.engine.forecast(data, self.origin)
        self.assertEqual(result['components']['wind']['status'], 'blocked')
        self.assertIsNone(result['components']['wind']['forecast_mw'])
        self.assertEqual(result['components']['demand']['status'], 'ok')
        self.assertEqual(result['components']['solar']['status'], 'ok')
        self.assertEqual(result['status'], 'degraded')
        for bad in [np.nan, -1., 'bad']:
            data = self.synthetic()
            data['demand_mw'] = data.demand_mw.astype(object)
            data.loc[data.index[-1], 'demand_mw'] = bad
            self.assertEqual(self.engine.forecast(data,self.origin)['components']['demand']['status'],'blocked')

    def test_frozen_detection_is_origin_safe_and_gaps_break_runs(self):
        data = self.synthetic(10)
        data.loc[5:9,'demand_mw'] = 100000.
        history = prepare_history(data,self.origin)
        self.assertFalse(quality_flags(history)[0]['demand'].frozen_reading.any())
        next_row = data.iloc[[-1]].copy(); next_row['Timestamp'] += pd.Timedelta(hours=1)
        with_future = pd.concat([data,next_row],ignore_index=True)
        self.assertFalse(quality_flags(prepare_history(with_future,self.origin))[0]['demand'].frozen_reading.any())
        flags = quality_flags(prepare_history(with_future,self.origin+pd.Timedelta(hours=1)))[0]
        self.assertEqual(flags['demand'].frozen_reading.sum(), 6)
        gappy = with_future.drop(index=7)
        self.assertFalse(quality_flags(prepare_history(gappy,self.origin+pd.Timedelta(hours=1)))[0]['demand'].frozen_reading.any())

    def test_zero_flatline_jump_and_component_inconsistency(self):
        data = self.synthetic()
        data.loc[data.index[-1],['demand_mw','wind_mw','solar_mw']] = 0.
        report = self.engine.forecast(data,self.origin)
        self.assertEqual(report['status'],'blocked')
        self.assertTrue(all(c['forecast_mw'] is None for c in report['components'].values()))
        data = self.synthetic()
        data.loc[data.index[-6:],'solar_mw'] = 64300.61
        report = self.engine.forecast(data,self.origin)
        self.assertEqual(report['components']['solar']['status'],'blocked')
        data = self.synthetic()
        data.loc[data.index[-1],'demand_mw'] *= .4
        report = self.engine.forecast(data,self.origin)
        self.assertIn('demand:sharp_demand_change',report['components']['demand']['reasons'])
        self.assertEqual(report['components']['demand']['status'],'blocked')
        data = self.synthetic(); data['solar_wind_mw'] = data.wind_mw + data.solar_mw
        data.loc[data.index[-1],'solar_wind_mw'] += 2
        report = self.engine.forecast(data,self.origin)
        self.assertEqual(report['components']['wind']['status'],'blocked')
        self.assertEqual(report['components']['solar']['status'],'blocked')
        self.assertEqual(report['components']['demand']['status'],'ok')

    def test_real_frozen_episode_requires_observed_evidence(self):
        early = self.engine.forecast(self.raw,'2025-01-08 23:00:00')
        later = self.engine.forecast(self.raw,'2025-01-09 02:00:00')
        self.assertEqual(early['components']['demand']['status'],'ok')
        self.assertEqual(later['components']['demand']['status'],'blocked')

    def test_january_fallback_and_missing_optional_checks(self):
        report = self.engine.forecast(self.raw,'2025-01-09 15:00:00')
        demand = report['components']['demand']
        self.assertEqual(demand['status'],'fallback')
        self.assertEqual(demand['forecast_mw'],203618.1)
        self.assertIn('demand:frozen_reading',demand['reasons'])
        report = self.engine.forecast(self.synthetic(),self.origin)
        self.assertIn('wind:renewable_component_balance_check_incomplete',report['components']['wind']['unavailable_quality_checks'])

    def test_training_cutoff(self):
        origin = '2024-05-05 15:00:00'
        data = self.synthetic(); data.Timestamp -= self.origin-pd.Timestamp(origin)
        report = self.engine.forecast(data,origin)
        for component in ['demand','wind']:
            self.assertEqual(report['components'][component]['status'],'fallback')
            self.assertIn(f'{component}:model_training_or_tuning_after_origin',report['components'][component]['reasons'])

    def test_invalid_history_schema(self):
        data = self.synthetic(169)
        for origin in ['2025-06-24 23:01:00','2025-06-24T23:00:00+05:30','NaT']:
            with self.assertRaises(HistoryError): self.engine.forecast(data,origin)
        duplicate = pd.concat([data,data.iloc[[-1]]])
        with self.assertRaises(HistoryError): self.engine.forecast(duplicate,self.origin)
        conflict = data.assign(timestamp=data.Timestamp+pd.Timedelta(hours=1))
        with self.assertRaises(HistoryError): self.engine.forecast(conflict,self.origin)
        agree = data.assign(timestamp=data.Timestamp)
        self.assertEqual(self.engine.forecast(agree,self.origin),self.engine.forecast(data,self.origin))
        with self.assertRaises(HistoryError): self.engine.forecast(data.iloc[:-1],self.origin)
        invalid = data.copy(); invalid.Timestamp=invalid.Timestamp.astype(object); invalid.loc[0,'Timestamp']='invalid'
        with self.assertRaises(HistoryError): self.engine.forecast(invalid,self.origin)

    def test_artifact_failures_fall_back_without_training(self):
        manifest = json.loads((ROOT/'models/forecast_manifest.json').read_text())
        manifest['components']['wind']['boosted_rounds'] += 1
        manifest['components']['demand']['artifact']['sha256'] = 'corrupt'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'; path.write_text(json.dumps(manifest))
            engine = GridForecaster(ROOT,manifest=path)
            result = engine.forecast(self.synthetic(),self.origin)['components']
            self.assertEqual(result['demand']['status'],'fallback')
            self.assertEqual(result['wind']['status'],'fallback')
            self.assertEqual(result['solar']['status'],'ok')
            manifest['components']['wind']['artifact']['path']='models/nonexistent.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(GridForecaster(ROOT,manifest=path).forecast(self.synthetic(),self.origin)['components']['wind']['status'],'fallback')

    def test_malformed_manifest_is_a_request_error(self):
        manifest = json.loads((ROOT/'models/forecast_manifest.json').read_text())
        manifest['components']['wind']['artifact'] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'; path.write_text(json.dumps(manifest))
            with self.assertRaises(HistoryError): GridForecaster(ROOT,manifest=path)

    def test_pure_features_never_include_future_labels(self):
        data = self.synthetic()
        past = prepare_history(data,self.origin)
        demand = demand_features(past,self.origin)
        wind = wind_features(past,self.origin)
        self.assertNotIn('Target',demand)
        self.assertNotIn('actual',wind)
        self.assertEqual(demand.demand_lag_1.iloc[0],past.demand_mw.iloc[-2])
        self.assertEqual(wind.prior_day.iloc[0],past.wind_mw.iloc[-1])
        self.assertAlmostEqual(wind.recent_mean_24.iloc[0],past.wind_mw.iloc[-24:].mean())
        self.assertAlmostEqual(wind.recent_std_168.iloc[0],past.wind_mw.iloc[-168:].std(ddof=1))


if __name__=='__main__': unittest.main()
