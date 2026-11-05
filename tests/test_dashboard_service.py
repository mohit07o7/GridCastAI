import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dashboard.service import (load_reports,load_history,load_forecaster,forecast,forecast_table,
    attach_actuals,metric_record,audit_slice,csv_bytes,json_bytes,signatures,NORMAL_ORIGIN,FALLBACK_ORIGIN,REPORT_DIR)


def forbidden(*a,**k):raise AssertionError('Training/battery solving must never run')


class DashboardServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports=load_reports(ROOT);cls.history=load_history(ROOT,cls.reports.summary)
        cls.model=load_forecaster(ROOT)
        with patch.object(xgb.XGBRegressor,'fit',forbidden),patch.object(xgb,'train',forbidden),patch('common.battery_dispatch.milp',forbidden):
            cls.normal=forecast(cls.model,cls.history,NORMAL_ORIGIN)

    def test_service_matches_cli_no_training(self):
        from forecasting.predict_grid_curve import main
        out=io.StringIO()
        argv=['predict_grid_curve','--origin',NORMAL_ORIGIN,'--history',str(ROOT/'data/processed/full_hourly_stacked.csv')]
        with patch.object(sys,'argv',argv),patch.object(xgb.XGBRegressor,'fit',forbidden),patch.object(xgb,'train',forbidden),patch('common.battery_dispatch.milp',forbidden),redirect_stdout(out):
            self.assertEqual(main(),0)
        expected=json.loads(out.getvalue())
        np.testing.assert_allclose(forecast_table(self.normal)[['demand_mw','wind_mw','solar_mw']],forecast_table(expected)[['demand_mw','wind_mw','solar_mw']],atol=1e-6,rtol=0)
        self.assertEqual(self.normal,expected)

    def test_future_values_never_change_forecast_or_quality(self):
        history=self.history.copy();origin=pd.Timestamp(FALLBACK_ORIGIN)
        a=forecast(self.model,history,origin)
        history.loc[history.timestamp>origin,['demand_mw','wind_mw','solar_mw']]=np.nan
        self.assertEqual(a,forecast(self.model,history,origin))

    def test_signed_quantities_zero_denominator_and_blocked_gaps(self):
        result=copy.deepcopy(self.normal)
        for p in result['points']:
            for c,v in [('demand',10.),('wind',8.),('solar',7.)]:p['components'][c]['forecast_mw']=v
        result['points'][1]['components']['demand']['forecast_mw']=0.
        result['points'][2]['components']['wind'].update(status='blocked',forecast_mw=None)
        table=forecast_table(result)
        self.assertEqual(table.iloc[0].net_demand_mw,-5.)
        self.assertEqual(table.iloc[0].renewable_share_pct,150.)
        self.assertTrue(np.isnan(table.iloc[1].renewable_share_pct))
        self.assertTrue(np.isnan(table.iloc[2].net_demand_mw))
        self.assertTrue(np.isnan(table.iloc[2].renewable_mw))

    def test_tables_downloads_actuals_and_alignment(self):
        table=forecast_table(self.normal)
        self.assertEqual(len(table),24)
        self.assertEqual(len(pd.read_csv(io.BytesIO(csv_bytes(table)))),24)
        self.assertEqual(json.loads(json_bytes(self.normal)),self.normal)
        actuals=attach_actuals(table,self.history)
        self.assertFalse(actuals.actual_available.any())
        self.assertTrue(actuals.target_quality.eq('unknown').all())
        wrong=copy.deepcopy(self.normal);wrong['points'][0]['target_timestamp']=NORMAL_ORIGIN
        with self.assertRaises(ValueError):forecast_table(wrong)

    def test_metric_selection_and_audit_counts(self):
        for h,month,cohort in [(24,'all','delivered'),(1,'2025-01','selected_only'),(12,'2025-06','delivered')]:
            row=metric_record(self.reports,'net',h,cohort,month)
            selected,counts,reasons=audit_slice(self.reports,h,month,cohort)
            self.assertEqual(row.rows,counts['scored_records'])
            self.assertEqual(row.nominal_rows,counts['forecast_records'])
            self.assertEqual(row.trusted_targets,counts['trusted_records'])
            self.assertEqual(counts['forecast_records'],counts['unique_target_hours'])
        with self.assertRaises(ValueError):metric_record(self.reports,horizon=25)

    def test_missing_models_preserve_reports_and_corrupt_reports_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);shutil.copytree(ROOT/REPORT_DIR,tmp/REPORT_DIR)
            self.assertEqual(len(load_reports(tmp).metrics),1728)
            with self.assertRaises(OSError):load_forecaster(tmp)
            file=tmp/REPORT_DIR/'forecast_metrics.csv';before=signatures([file])
            file.write_text(file.read_text()+'\n')
            self.assertNotEqual(before,signatures([file]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_reports(tmp)


if __name__=='__main__':unittest.main()
