import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.features import build_features
from common.splits import chronological_split

class DemandPipelineTests(unittest.TestCase):
    def data(self):
        n = 1000
        return pd.DataFrame({'Timestamp': pd.date_range('2024-01-01', periods=n, freq='h'), 'demand_mw': np.arange(n) + 100., 'wind_mw': np.ones(n), 'solar_mw': np.ones(n)})

    def test_origin_features_and_future_invariance(self):
        raw = self.data()
        f = build_features(raw)
        self.assertEqual(f.loc[200, 'Target'], 324.)
        self.assertEqual(f.loc[200, 'demand_lag_24'], 276.)
        self.assertAlmostEqual(f.loc[200, 'demand_mean_24h'], raw.demand_mw.iloc[176:200].mean())
        raw.loc[201:, 'demand_mw'] += 1000
        other = build_features(raw)
        for col in [c for c in f if '_lag_' in c or '_mean_' in c or '_trend_' in c]:
            self.assertEqual(f.loc[200,col], other.loc[200,col])

    def test_policy_window_edges_and_target(self):
        raw = self.data()
        raw.loc[100, ['demand_mw','wind_mw','solar_mw']] = 0
        f = build_features(raw)
        self.assertTrue(f.loc[76, 'quality_target_bad'])
        self.assertTrue(f.loc[268, 'quality_input_bad'])
        self.assertFalse(f.loc[269, 'quality_input_bad'])
        raw = self.data()
        raw.loc[300:305, 'demand_mw'] = 10
        self.assertEqual(build_features(raw).demand_frozen_flag.sum(), 6)

    def test_purge_and_invalid_timestamps(self):
        f = build_features(self.data()).dropna()
        a,b,c = chronological_split(f)
        self.assertLess(a['Target Timestamp'].max(), b.Timestamp.min())
        self.assertLess(b['Target Timestamp'].max(), c.Timestamp.min())
        bad = self.data(); bad.loc[10,'Timestamp'] = bad.loc[9,'Timestamp']
        with self.assertRaises(ValueError): build_features(bad)

if __name__ == '__main__': unittest.main()
