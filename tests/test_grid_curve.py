import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common.curve_features import (BASELINES, COMPONENTS, FEATURES, FEATURE_VERSION,
    POLICY, HOUR, VALIDATION_START, TEST_START, CurveHistory, split_masks)
from common.curve_model import GridCurveForecaster, METHOD, digest, fingerprint, json_write


def synthetic(n=400,start='2025-01-01'):
    i=np.arange(n,dtype=float)
    return pd.DataFrame({'timestamp':pd.date_range(start,periods=n,freq='h'),
        'demand_mw':100+5*np.sin(i/10)+i/100,
        'wind_mw':10+np.sin(i/11)+i/1000,
        'solar_mw':np.maximum(0,20*np.sin(2*np.pi*(i%24-6)/24))})


def make_bundle(path,data):
    history=CurveHistory(data)
    features=history.features('demand',1,history.index[-40:])
    model=XGBRegressor(n_estimators=3,early_stopping_rounds=2,max_depth=2,
        random_state=42,n_jobs=1,objective='reg:absoluteerror')
    model.fit(features,np.ones(len(features)),eval_set=[(features,np.ones(len(features)))],verbose=False)
    entries={}
    for c in COMPONENTS:
        for h in range(1,25):
            key=f'{c}_h{h:02d}'
            entry={'component':c,'horizon':h,'model_file':key+'.json',
                'best_iteration':int(model.best_iteration),
                'boosted_rounds':model.get_booster().num_boosted_rounds(),
                'selected_method':METHOD,'baseline_ranking':list(BASELINES),
                'available_after_target':'2024-11-28 00:00:00'}
            model.get_booster().set_attr(gridcast_curve_metadata_sha256=fingerprint(entry))
            model.save_model(path/entry['model_file'])
            entry['model_sha256']=digest(path/entry['model_file'])
            entries[key]=entry
    json_write(path/'metadata.json',{'schema_version':1,'model_id':'grid_curve_v1',
        'features':[{'name':x,'dtype':'float64'} for x in FEATURES],
        'horizons':list(range(1,25)),'feature_version':FEATURE_VERSION,
        'inference_policy':POLICY,'entries':entries})


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.tmp.name)
        cls.data=synthetic()
        make_bundle(cls.path,cls.data)
        cls.forecaster=GridCurveForecaster(ROOT,cls.path,strict=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_features_manual_and_horizon_baselines(self):
        context=CurveHistory(self.data)
        origin=context.index[-1]
        for h in [1,12,24]:
            for c in COMPONENTS:
                f=context.features(c,h,[origin]).iloc[0]
                v=self.data[f'{c}_mw']
                self.assertEqual(f.lag_168,v.iloc[-169])
                self.assertAlmostEqual(f.recent_mean_24,v.iloc[-24:].mean())
                self.assertAlmostEqual(f.recent_std_168,v.iloc[-168:].std(ddof=1))
                self.assertEqual(f.target_previous_day,v.iloc[-1-(24-h)])
                self.assertEqual(f.target_previous_week,v.iloc[-1-(168-h)])
                self.assertEqual(context.baseline(c,'last_observation',h,[origin])[0],v.iloc[-1])

    def test_no_future_influence_and_exactly_24_targets(self):
        origin=self.data.timestamp.iloc[-25]
        a=self.forecaster.forecast(self.data,origin)
        future=self.data.copy()
        future.loc[future.timestamp>origin,['demand_mw','wind_mw','solar_mw']]=np.nan
        self.assertEqual(a,self.forecaster.forecast(future,origin))
        self.assertEqual(a,self.forecaster.forecast(self.data[self.data.timestamp<=origin],origin))
        self.assertEqual([p['horizon'] for p in a['points']],list(range(1,25)))
        self.assertEqual([p['target_timestamp'] for p in a['points']],
                         [str(origin+h*HOUR) for h in range(1,25)])

    def test_batch_and_prefix_feature_quality_parity(self):
        data=self.data.copy()
        data.loc[260:270,'wind_mw']=15.
        full=CurveHistory(data)
        for i in [259,260,264,265,270,280]:
            origin=data.timestamp.iloc[i]
            prefix=CurveHistory(data,origin)
            for h in [1,24]:
                pd.testing.assert_frame_equal(full.features('wind',h,[origin]),prefix.features('wind',h,[origin]))
                for offset in [None,0,4,10]:
                    pd.testing.assert_frame_equal(full.quality('wind',[origin],offset),prefix.quality('wind',[origin],offset))
        self.assertFalse(full.quality('wind',[data.timestamp.iloc[264]],0).frozen_reading.iloc[0])
        self.assertTrue(full.quality('wind',[data.timestamp.iloc[265]],0).frozen_reading.iloc[0])
        self.assertTrue(full.quality('wind',[data.timestamp.iloc[265]],5).frozen_reading.iloc[0])

    def test_gaps_invalid_values_and_component_isolation(self):
        origin=self.data.timestamp.iloc[-1]
        data=self.data.drop(index=len(self.data)-10)
        result=self.forecaster.forecast(data,origin)
        self.assertEqual(result['points'][0]['components']['demand']['status'],'fallback')
        data=self.data.copy()
        data.loc[len(data)-169:,'wind_mw']=-1
        result=self.forecaster.forecast(data,origin)
        self.assertEqual(result['points'][0]['components']['wind']['status'],'blocked')
        self.assertEqual(result['points'][0]['components']['demand']['status'],'ok')
        self.assertFalse(result['complete_curve'])

    def test_alternate_schema_shuffle_and_bad_timestamps(self):
        origin=self.data.timestamp.iloc[-1]
        a=self.forecaster.forecast(self.data,origin)
        self.assertEqual(a,self.forecaster.forecast(self.data.rename(columns={'timestamp':'Timestamp'}).sample(frac=1,random_state=42),origin))
        with self.assertRaises(ValueError):self.forecaster.forecast(self.data,'2025-01-02 00:30')
        with self.assertRaises(ValueError):self.forecaster.forecast(pd.concat([self.data,self.data.iloc[-1:]]),origin)

    def test_metadata_corruption_and_fresh_load(self):
        original=(self.path/'metadata.json').read_text()
        try:
            meta=json.loads(original)
            meta['entries']['demand_h01']['selected_method']='previous_day'
            json_write(self.path/'metadata.json',meta)
            with self.assertRaises(ValueError):GridCurveForecaster(ROOT,self.path,strict=True)
        finally:(self.path/'metadata.json').write_text(original)
        fresh=GridCurveForecaster(ROOT,self.path,strict=True)
        self.assertEqual(self.forecaster.forecast(self.data,self.data.timestamp.iloc[-1]),fresh.forecast(self.data,self.data.timestamp.iloc[-1]))

    def test_purge_and_model_availability(self):
        origins=pd.date_range(VALIDATION_START-48*HOUR,TEST_START+48*HOUR,freq='h')
        m=split_masks(origins)
        self.assertLess((origins[m['train']]+24*HOUR).max(),origins[m['validation']].min())
        self.assertLess((origins[m['validation']]+24*HOUR).max(),origins[m['test']].min())
        data=synthetic(start='2023-01-01')
        result=self.forecaster.forecast(data,data.timestamp.iloc[-1])
        self.assertTrue(all(p['components']['demand']['status']=='fallback' for p in result['points']))
        self.assertTrue(all(p['target_quality']=='unknown' for p in result['points']))


if __name__=='__main__':unittest.main()
