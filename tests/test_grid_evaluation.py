import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.grid_evaluation import (
    validate_component, validate_keys, combined_quantities, assign_replay_cohorts,
    paired_summary, attach_observations, retrospective_target_flags, attach_target_flags,
)
from analysis.evaluate_grid_forecasts import replay_predictions


class GridEvaluationTests(unittest.TestCase):
    def source(self):
        t=pd.date_range('2025-04-03',periods=96,freq='h');n=np.arange(96,dtype=float)
        return pd.DataFrame({'demand_mw':200000+n,'wind_mw':1000+n,'solar_mw':200+n,
                             'solar_wind_mw':1200+2*n,'total_gen_mw':210000+n},index=t)

    def frame(self):
        return pd.DataFrame({'origin':pd.to_datetime(['2025-04-03 00:00','2025-04-03 01:00']),
            'target_timestamp':pd.to_datetime(['2025-04-04 00:00','2025-04-04 01:00']),
            'actual_demand':[100.,100.],'actual_wind':[30.,30.],'actual_solar':[80.,80.],
            'predicted_demand':[110.,110.],'predicted_wind':[40.,40.],'predicted_solar':[90.,90.],
            'baseline_demand':[120.,120.],'baseline_wind':[10.,10.],'baseline_solar':[20.,20.],
            'trusted_target':[True,True], 'demand_status':['ok','fallback'],
            'wind_status':['ok','ok'], 'solar_status':['ok','ok']})

    def test_keys_actual_and_persistence_alignment(self):
        source=self.source(); f=self.frame()[['origin','target_timestamp']]
        f=attach_observations(f,source); f['predicted_demand']=f.actual_demand
        validate_component(f,source,'demand')
        duplicate=pd.concat([f,f.iloc[[0]]])
        with self.assertRaisesRegex(ValueError,'duplicate'): validate_keys(duplicate,'test')
        shifted=f.copy(); shifted.target_timestamp+=pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError,'24h'): validate_keys(shifted,'test')
        for column in ['actual_demand','baseline_demand']:
            bad=f.copy();bad.loc[0,column]+=1
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError,'timestamp alignment'):validate_component(bad,source,'demand')

    def test_hand_calculated_signed_quantities(self):
        f=combined_quantities(self.frame())
        self.assertEqual(f.actual_net_demand_mw.iloc[0],-10)
        self.assertEqual(f.predicted_net_demand_mw.iloc[0],-20)
        self.assertEqual(f.baseline_net_demand_mw.iloc[0],90)
        self.assertEqual(f.actual_renewable_mw.iloc[0],110)
        self.assertAlmostEqual(f.actual_renewable_share_pct.iloc[0],110)
        self.assertEqual(f.predicted_net_error_mw.iloc[0],-10)
        self.assertEqual(f.baseline_net_error_mw.iloc[0],100)
        result=paired_summary(f,pd.Series([True,True]),'example',2,2)
        self.assertEqual(result['predicted_net_mae_mw'],10)
        self.assertEqual(result['baseline_net_mae_mw'],100)
        self.assertEqual(result['mae_improvement_pct'],90)
        self.assertEqual(result['predicted_net_mean_signed_error_mw'],-10)
        self.assertEqual(result['bootstrap']['mae_improvement_mw'],90)

    def test_zero_denominators_and_empty_cohort(self):
        f=self.frame(); f['actual_demand']=0.; f['predicted_demand']=0.
        f=combined_quantities(f)
        result=paired_summary(f,pd.Series([True,True]),'zero_share',2,2)
        self.assertEqual(result['paired_share_rows'],0)
        self.assertIsNone(result['predicted_renewable_share_mae_pp'])
        self.assertTrue(f.actual_share_undefined.all())
        f=self.frame();f['actual_demand']=110.;f['baseline_demand']=110.;f['baseline_wind']=30.;f['baseline_solar']=80.
        f=combined_quantities(f);result=paired_summary(f,pd.Series([True,True]),'zero_net',2,2)
        self.assertIsNone(result['predicted_net_wape_pct'])
        self.assertIsNone(result['mae_improvement_pct'])
        empty=paired_summary(f,pd.Series([False,False]),'empty',2,0)
        self.assertEqual(empty['rows'],0)
        self.assertIsNone(empty['predicted_net_mae_mw'])
        self.assertIsNone(empty['bootstrap']['ci95_low_mw'])
        self.assertIsNone(empty['coverage_trusted_pct'])

    def test_same_scoring_rows_and_fallback_cohorts(self):
        f=assign_replay_cohorts(self.frame())
        self.assertEqual(f.score_selected_only.tolist(),[True,False])
        self.assertEqual(f.score_delivered.tolist(),[True,True])
        f.loc[1,'trusted_target']=False
        f=assign_replay_cohorts(f)
        self.assertEqual(f.score_delivered.tolist(),[True,False])
        f=combined_quantities(f)
        # Deliberately make the excluded baseline awful; it must not change its score.
        f.loc[1,'baseline_net_demand_mw']=1e12
        result=paired_summary(f,f.score_delivered,'delivered',2,1)
        self.assertEqual(result['rows'],1)
        self.assertEqual(result['baseline_net_mae_mw'],100)
        self.assertEqual(result['predicted_net_mae_mw'],10)

    def test_missing_blocked_and_nonfinite_pairs(self):
        f=self.frame();f.loc[1,'predicted_wind']=np.nan;f.loc[1,'wind_status']='blocked'
        f=assign_replay_cohorts(f)
        self.assertEqual(f.score_delivered.tolist(),[True,False])
        self.assertFalse(f.any_fallback.iloc[1] and f.score_delivered.iloc[1])
        combined=combined_quantities(f)
        with self.assertRaisesRegex(ValueError,'nonfinite paired'):paired_summary(combined,pd.Series([True,True]),'bad',2,2)
        bad=self.frame();bad.loc[0,'wind_status']='blocked'
        with self.assertRaises(ValueError):assign_replay_cohorts(bad)
        bad=self.frame();bad.loc[0,'baseline_wind']=np.nan
        self.assertFalse(assign_replay_cohorts(bad).score_delivered.iloc[0])

    def test_target_audit_is_separate_from_prediction(self):
        source=self.source(); origin=pd.Timestamp('2025-04-03 16:00')
        nominal=pd.DataFrame({'origin':[origin],'target_timestamp':[origin+pd.Timedelta(hours=24)]})
        class Stub:
            def forecast(self,history,stamp):
                return {'origin':str(stamp),'target_timestamp':str(stamp+pd.Timedelta(hours=24)),
                    'status':'ok','target_quality':'unknown','run_context':'simulation',
                    'components':{c:{'target_timestamp':str(stamp+pd.Timedelta(hours=24)), 'forecast_mw':v,
                        'status':'ok','method':'test','selected_model_id':'test','selected_artifact_sha256':None,
                        'reasons':[],'unavailable_quality_checks':[]} for c,v in [('demand',200000.),('wind',1000.),('solar',200.)]}}
        predictions=replay_predictions(source.reset_index(),nominal,Stub())
        self.assertNotIn('actual_demand',predictions)
        self.assertNotIn('trusted_target',predictions)
        audited=attach_target_flags(attach_observations(predictions,source),retrospective_target_flags(source))
        self.assertFalse(audited.trusted_target.iloc[0])
        self.assertTrue(audited.target_april_anomaly.iloc[0])
        self.assertEqual(audited.demand_status.iloc[0],'ok')
        self.assertEqual(audited.inference_target_quality.iloc[0],'unknown')
        self.assertFalse(assign_replay_cohorts(audited).score_delivered.iloc[0])

    def test_target_flag_definitions(self):
        source=self.source()
        source.loc[source.index[10:16],'wind_mw']=123.
        source.loc[source.index[20:26],'solar_mw']=64300.61
        source.loc[source.index[30],['demand_mw','wind_mw','solar_mw']]=0.
        flags=retrospective_target_flags(source)
        self.assertEqual(flags.target_frozen_wind.sum(),6)
        self.assertEqual(flags.target_positive_solar_flatline.sum(),6)
        self.assertTrue(flags.target_all_zero.iloc[30])
        self.assertTrue(flags.target_nonpositive_demand.iloc[30])
        self.assertTrue(flags.target_renewable_inconsistency.iloc[20])
        source.loc[source.index[0],'solar_mw']=np.nan
        self.assertTrue(retrospective_target_flags(source).target_invalid_components.iloc[0])


if __name__=='__main__':unittest.main()
