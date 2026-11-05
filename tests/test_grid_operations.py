from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.battery_dispatch import Scenario
from common.grid_operations import attach_evaluation,forecast_scores,simulate,operations_scores


def curves(origins):
    rows=[]
    for o in origins:
        for h in range(1,25):
            row={'origin':o,'target_timestamp':o+pd.Timedelta(hours=h),'horizon':h}
            for c,value in [('demand',100.),('wind',10.),('solar',5.)]:
                row.update({c+'_mw':value,c+'_status':'ok',c+'_method':'previous_day',
                            c+'_reasons':'[]',c+'_unavailable_checks':'[]'})
            row['net_demand_mw']=85.;row['renewable_mw']=15.
            rows.append(row)
    return pd.DataFrame(rows)


class OperationsTests(unittest.TestCase):
    def data(self):
        origins=pd.date_range('2025-04-04 14:00',periods=3,freq='h')
        index=pd.date_range(origins[0],origins[-1]+pd.Timedelta(hours=24),freq='h')
        i=np.arange(len(index))
        source=pd.DataFrame({'demand_mw':100+i/100,'wind_mw':10+i/1000,
                             'solar_mw':5+i/1000},index=index)
        return curves(origins),source

    def test_alignment_quality_and_paired_masks(self):
        selected,source=self.data();baseline=selected.copy()
        selected.loc[0,['solar_mw','net_demand_mw']]=np.nan
        selected.loc[0,'solar_status']='blocked'
        result=attach_evaluation(selected,baseline,source)
        april=result.target_timestamp.eq(pd.Timestamp('2025-04-04 16:00'))
        self.assertFalse(result.loc[april,'trusted_target'].any())
        self.assertFalse(result.iloc[0].score_delivered)
        result.loc[result.index[1],'demand_status']='fallback'
        # Masks are attached from actual status, not inferred from the method name.
        selected.loc[1,'demand_status']='fallback'
        result=attach_evaluation(selected,baseline,source)
        self.assertTrue(result.iloc[1].score_delivered if result.iloc[1].trusted_target else True)
        self.assertFalse(result.iloc[1].score_selected_only)
        scores=forecast_scores(result)
        self.assertTrue((scores.rows<=scores.nominal_rows).all())
        duplicate=pd.concat([baseline,baseline.iloc[:1]])
        with self.assertRaises(ValueError):attach_evaluation(selected,duplicate,source)
        shifted=baseline.copy();shifted.loc[0,'target_timestamp']+=pd.Timedelta(hours=1)
        with self.assertRaises(ValueError):attach_evaluation(selected,shifted,source)

    def test_plans_precede_target_exclusions_and_episode_reset(self):
        selected,source=self.data()
        s=replace(Scenario(),scale_factor=.1,training_peak_mw=1000.)
        calls=[]
        class FakeOptimizer:
            def __init__(self,scenario):pass
            def solve(self,curve,soc):
                calls.append((curve.origin.iloc[0],soc))
                return {'status':'optimal','reasons':[],
                    'hours':pd.DataFrame({'charge_mw':[0.]*24,'discharge_mw':[0.]*24}),
                    'predicted_unmet_mwh':0.,'predicted_electricity_cost_inr':1.,'predicted_wear_cost_inr':0.}
        with patch('common.grid_operations.DispatchOptimizer',FakeOptimizer):
            result=simulate(selected,selected.copy(),source,s)
        self.assertEqual(len(calls),6) # Includes both plans for the excluded April 4 target.
        self.assertEqual(len(result),9)
        self.assertEqual(int(result.trusted_target.sum()),6)
        self.assertEqual(int(result.episode_start.sum()),6)
        self.assertEqual(len(result[result.policy.eq('forecast_battery') & result.trusted_target]),2)

    def test_blocked_plan_idles_without_withholding_actual_hours(self):
        selected,source=self.data();baseline=selected.copy()
        selected.loc[0,'solar_mw']=np.nan
        selected.loc[0,'solar_status']='blocked'
        s=replace(Scenario(),scale_factor=.1,training_peak_mw=1000.)
        result=simulate(selected,baseline,source,s)
        row=result[result.policy.eq('forecast_battery')].iloc[0]
        self.assertTrue(row.trusted_target)
        self.assertEqual(row.plan_status,'unavailable')
        self.assertEqual(row.charge_mw,0)
        self.assertEqual(row.discharge_mw,0)
        metrics=operations_scores(result)
        counts=metrics[metrics.month.eq('all')].trusted_hours
        self.assertEqual(counts.nunique(),1)


if __name__=='__main__':unittest.main()
