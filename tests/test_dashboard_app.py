from datetime import date
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dashboard.service import load_reports,metric_record


def forbidden(*a,**k):raise AssertionError('No training or battery solving in the UI')


class DashboardAppTests(unittest.TestCase):
    def setUp(self):
        self.guards=[patch.object(xgb.XGBRegressor,'fit',forbidden),patch.object(xgb,'train',forbidden),
                     patch('common.battery_dispatch.milp',forbidden)]
        for guard in self.guards:guard.start();self.addCleanup(guard.stop)

    def app(self):
        app=AppTest.from_file(str(ROOT/'src/dashboard/app.py')).run(timeout=60)
        self.assertFalse(app.exception,list(app.exception))
        return app

    def test_default_filter_change_and_retained_forecast(self):
        app=self.app()
        result=app.session_state['result']['forecast']
        self.assertEqual(len(result['points']),24)
        self.assertEqual(result['origin'],'2025-06-24 23:00:00')
        app.selectbox(key='horizon').set_value(1).run(timeout=60)
        expected=metric_record(load_reports(ROOT),horizon=1)
        value=next(m.value for m in app.metric if m.label=='Forecast MAE')
        self.assertEqual(value,f'{expected.forecast_mae_mw:,.2f} MW')
        self.assertEqual(app.session_state['result']['forecast'],result)
        app.date_input(key='issue_date').set_value(date(2025,6,23)).run(timeout=60)
        self.assertEqual(app.session_state['result']['forecast'],result)
        self.assertFalse(app.exception)

    def test_presets_actual_overlay_and_deferred_status(self):
        app=self.app()
        next(b for b in app.button if b.label=='Historical fallback').click().run(timeout=60)
        self.assertEqual(app.session_state['result']['forecast']['status'],'degraded')
        app.checkbox(key='show_actual').check().run(timeout=60)
        self.assertFalse(app.exception,list(app.exception))
        next(b for b in app.button if b.label=='Synthetic blocked wind').click().run(timeout=60)
        result=app.session_state['result']['forecast']
        self.assertFalse(result['complete_curve'])
        self.assertTrue(all(p['components']['wind']['forecast_mw'] is None for p in result['points']))
        self.assertTrue(any('Deferred' in h.value for h in app.subheader))
        self.assertFalse(app.exception,list(app.exception))

    def test_missing_models_keep_accuracy_accessible(self):
        st.cache_resource.clear()
        try:
            with patch('dashboard.service.load_forecaster',side_effect=FileNotFoundError('model fixture missing')):
                app=self.app()
                self.assertTrue(any('New forecasts unavailable' in w.value for w in app.warning))
                self.assertTrue(any(m.label=='Forecast MAE' for m in app.metric))
        finally:st.cache_resource.clear()


if __name__=='__main__':unittest.main()
