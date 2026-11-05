from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.battery_dispatch import (Scenario,DispatchOptimizer,optimize_dispatch,
    execute_hour,validate_dispatch,scenario_variants)


def curve(demand=10.,wind=0.,solar=0.):
    return pd.DataFrame({'horizon':range(1,25),
        'target_timestamp':pd.date_range('2025-01-01',periods=24,freq='h'),
        'demand_mw':np.broadcast_to(demand,24),'wind_mw':np.broadcast_to(wind,24),
        'solar_mw':np.broadcast_to(solar,24)})


class DispatchTests(unittest.TestCase):
    def scenario(self,**kwargs):return replace(Scenario(),scale_factor=1.,training_peak_mw=100.,**kwargs)

    def test_zero_battery_hand_calculated_balance_and_cost(self):
        s=self.scenario(battery_capacity_mwh=0.)
        result=optimize_dispatch(curve(10,2,3),s)
        self.assertEqual(result['status'],'optimal')
        h=result['hours']
        np.testing.assert_allclose(h.grid_mw,5,atol=1e-6)
        np.testing.assert_allclose(h.renewable_used_mw,5,atol=1e-6)
        self.assertAlmostEqual(result['predicted_electricity_cost_inr'],5*sum(s.hourly_prices_inr_per_mwh),places=3)
        self.assertLess(result['predicted_unmet_mwh'],1e-5)
        self.assertTrue(validate_dispatch(h,s,0))

    def test_shortage_is_reported_and_never_fully_served(self):
        s=self.scenario(grid_limit_mw=5.,battery_capacity_mwh=0.)
        result=optimize_dispatch(curve(10,1,1),s)
        self.assertFalse(result['fully_served'])
        self.assertAlmostEqual(result['minimum_unmet_mwh'],72.,places=5)
        self.assertAlmostEqual(result['predicted_unmet_mwh'],72.,places=4)

    def test_surplus_curtailment_and_no_export(self):
        s=self.scenario(battery_capacity_mwh=0.)
        result=optimize_dispatch(curve(5,4,4),s)
        self.assertAlmostEqual(result['hours'].curtailment_mw.sum(),72.,places=4)
        self.assertGreaterEqual(result['hours'].grid_mw.min(),-1e-6)

    def test_efficiency_terminal_reserve_and_arbitrage(self):
        s=self.scenario()
        result=optimize_dispatch(curve(50),s)
        self.assertEqual(result['status'],'optimal')
        h=result['hours']
        self.assertGreater(h.charge_mw.sum(),0)
        self.assertGreater(h.discharge_mw.sum(),0)
        self.assertLess(h.discharge_mw.sum(),h.charge_mw.sum())
        self.assertGreaterEqual(h.soc_end_mwh.iloc[-1],40-1e-6)
        self.assertLess(result['predicted_electricity_cost_inr']+result['predicted_wear_cost_inr'],50*sum(s.hourly_prices_inr_per_mwh))
        self.assertTrue(validate_dispatch(h,s,40))
        broken=h.copy();broken.loc[0,'grid_mw']+=1
        with self.assertRaises(ValueError):validate_dispatch(broken,s,40)

    def test_reliability_has_priority_over_price(self):
        s=self.scenario(grid_limit_mw=50.)
        demand=np.full(24,40.);demand[18:20]=65.
        result=optimize_dispatch(curve(demand),s)
        self.assertEqual(result['status'],'optimal')
        self.assertLess(result['predicted_unmet_mwh'],1e-5)
        self.assertGreater(result['hours'].discharge_mw.iloc[18:20].sum(),29.99)

    def test_executed_action_clamps_and_state_carry(self):
        s=self.scenario(grid_limit_mw=10.)
        a=execute_hour(12,0,0,5,0,40,'2025-01-01',s)
        self.assertEqual(a['charge_mw'],0)
        self.assertEqual(a['unmet_mw'],2)
        self.assertTrue(a['action_adjusted'])
        b=execute_hour(10,1,1,0,3,a['soc_end_mwh'],'2025-01-01 01:00',s)
        self.assertAlmostEqual(b['soc_end_mwh'],40-3/s.discharge_efficiency)
        c=execute_hour(10,20,0,0,5,b['soc_end_mwh'],'2025-01-01 02:00',s)
        self.assertEqual(c['discharge_mw'],0)
        self.assertEqual(c['soc_start_mwh'],b['soc_end_mwh'])

    def test_invalid_incomplete_curve_and_solver_failure(self):
        s=self.scenario()
        bad=curve();bad.loc[0,'solar_mw']=np.nan
        self.assertEqual(optimize_dispatch(bad,s)['status'],'unavailable')
        with self.assertRaises(ValueError):optimize_dispatch(curve(),s,1000)
        with patch('common.battery_dispatch.milp',side_effect=RuntimeError('test failure')):
            self.assertEqual(optimize_dispatch(curve(),s)['status'],'unavailable')
        with self.assertRaises(ValueError):self.scenario(charge_efficiency=0.)
        variants=scenario_variants(s)
        self.assertEqual([v.initial_soc_mwh for v in variants[-2:]],[20.,80.])


if __name__=='__main__':unittest.main()
