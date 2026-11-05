import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from starlette.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from command_api.app import create_app,ResourceUnavailable
from services.grid_service import forecast,forecast_table,metric_record


class CommandApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgboost as xgb
        import common.battery_dispatch as battery
        cls.original_entrypoints=[(owner,name,getattr(owner,name)) for owner,name in
            [(xgb.XGBRegressor,'fit'),(xgb,'train'),(xgb.Booster,'update'),(battery,'milp'),(battery,'optimize_dispatch')]]
        cls.app=create_app(ROOT)
        cls.client=TestClient(cls.app)
        cls.store=cls.app.state.resources

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        for owner,name,value in cls.original_entrypoints:setattr(owner,name,value)

    def test_fresh_api_without_retired_interface_imports(self):
        # Fail on attempted imports even when Streamlit remains installed locally.
        script = """
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
class NoRetiredInterface(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'streamlit', 'dashboard', 'plotly'}:
            raise AssertionError('Retired UI import: ' + fullname)
sys.meta_path.insert(0, NoRetiredInterface())
from command_api.app import create_app
from starlette.testclient import TestClient
with TestClient(create_app()) as client:
    assert client.get('/api/v1/history?limit=1').status_code == 200
    result = client.post('/api/v1/forecast', json={'origin': '2025-06-24 23:00:00'})
    assert result.status_code == 200, result.text
    assert len(result.json()['forecast']['points']) == 24
from services.runtime import disabled
import xgboost as xgb
assert xgb.XGBRegressor.fit is disabled
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_forecast_parity_and_future_invariance(self):
        origin='2025-01-09 15:00:00'
        response=self.client.post('/api/v1/forecast',json={'origin':origin})
        self.assertEqual(response.status_code,200,response.text)
        expected=forecast(self.store.get('models'),self.store.get('history'),origin)
        self.assertEqual(response.json()['forecast'],expected)
        table=forecast_table(expected)
        np.testing.assert_allclose(np.array([[p[c+'_mw'] for c in ['demand','wind','solar']] for p in response.json()['table']]),table[['demand_mw','wind_mw','solar_mw']],atol=1e-6,rtol=0)
        signature,original=self.store.cache['history']
        modified=original.copy();modified.loc[modified.timestamp>origin,['demand_mw','wind_mw','solar_mw']]=np.nan
        try:
            self.store.cache['history']=(signature,modified)
            self.assertEqual(response.json()['forecast'],self.client.post('/api/v1/forecast',json={'origin':origin}).json()['forecast'])
        finally:self.store.cache['history']=(signature,original)

    def test_metrics_mape_and_error_distribution(self):
        result=self.client.get('/api/v1/metrics?quantity=solar&horizon=24&cohort=delivered&month=all')
        self.assertEqual(result.status_code,200,result.text);value=result.json()
        reports=self.store.get('reports');saved=metric_record(reports,'solar',24,'delivered','all')
        self.assertEqual(saved.rows,value['saved']['rows'])
        selected=reports.rows[(reports.rows.horizon==24)&reports.rows.score_delivered]
        mask=selected.actual_solar_mw.ne(0)
        mape=100*((selected.loc[mask,'solar_mw']-selected.loc[mask,'actual_solar_mw']).abs()/selected.loc[mask,'actual_solar_mw'].abs()).mean()
        self.assertAlmostEqual(value['mape']['forecast_pct'],mape)
        self.assertEqual(value['mape']['paired_nonzero_actual_rows'],int(mask.sum()))
        self.assertEqual(sum(value['error_histogram']['counts']),int(saved.rows))
        self.assertEqual(value['counts']['scored_unique_target_hours'],saved.rows)

    def test_feature_gain_and_model_metadata(self):
        result=self.client.get('/api/v1/models?component=wind&horizon=6')
        self.assertEqual(result.status_code,200,result.text)
        model=self.store.get('models').models['wind_h06']
        native=model.get_booster()[:model.best_iteration+1].get_score(importance_type='gain')
        self.assertEqual(result.json()['importance_label'],'XGBoost split-gain importance')
        for row in result.json()['feature_importance']:self.assertEqual(row['gain'],native.get(row['feature'],0))

    def test_history_pagination_and_download_full_resolution(self):
        url='/api/v1/history?start=2025-06-01%2000:00:00&end=2025-06-24%2023:00:00'
        result=self.client.get(url+'&limit=10').json()
        self.assertEqual(len(result['rows']),10);self.assertEqual(result['total'],24*24)
        csv=self.client.get(url+'&limit=10&format=csv')
        self.assertEqual(csv.status_code,200);self.assertEqual(len(csv.text.splitlines()),24*24+1)
        self.assertIn('retrospective_bad',csv.text)
        self.assertEqual(self.client.get(url+'&limit=5001').status_code,422)

    def test_blocked_demo_and_invalid_requests(self):
        self.assertEqual(self.client.post('/api/v1/forecast',json={'origin':'2025-06-24 23:00:00','case':'blocked'}).status_code,422)
        result=self.client.post('/api/v1/forecast',json={'origin':'2025-06-24 23:00:00','case':'blocked','demo':True}).json()
        self.assertTrue(result['synthetic']);self.assertFalse(result['forecast']['complete_curve'])
        self.assertTrue(all(p['wind_mw'] is None and p['net_demand_mw'] is None for p in result['table']))
        self.assertEqual(self.client.post('/api/v1/forecast',json={'origin':'2025-06-24 23:30'}).status_code,422)
        self.assertEqual(self.client.get('/api/v1/metrics?horizon=25').status_code,422)
        self.assertEqual(self.client.get('/api/v1/unknown').status_code,404)

    def test_model_outage_preserves_history_and_actual_request_logs(self):
        original=self.store.get
        def unavailable(name):
            if name=='models':raise ResourceUnavailable('models unavailable: test fixture')
            return original(name)
        with patch.object(self.store,'get',side_effect=unavailable):
            self.assertEqual(self.client.post('/api/v1/forecast',json={'origin':'2025-06-24 23:00:00'}).status_code,503)
            self.assertEqual(self.client.get('/api/v1/metrics').status_code,200)
            self.assertEqual(self.client.get('/api/v1/history').status_code,200)
            self.assertFalse(self.client.get('/api/v1/status').json()['resources']['models']['available'])
        logs=self.client.get('/api/v1/logs').json()['rows']
        self.assertTrue(any(e['status']==503 for e in logs))
        self.assertTrue(all(e['duration_ms']>=0 and e['time_utc'] for e in logs))


if __name__=='__main__':unittest.main()
