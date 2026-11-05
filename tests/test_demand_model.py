import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.demand_model import save_demand_bundle,load_demand_bundle


class DemandBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bundle=self.root/'bundle'
        self.x=pd.DataFrame({'lag':np.arange(40,dtype=float),'hour':np.arange(40)%24})
        self.y=self.x.lag*2+self.x.hour
        self.model=XGBRegressor(n_estimators=12,max_depth=2,early_stopping_rounds=3,random_state=42,n_jobs=1)
        self.model.fit(self.x.iloc[:30],self.y.iloc[:30],eval_set=[(self.x.iloc[30:],self.y.iloc[30:])],verbose=False)
        times=pd.date_range('2024-01-01',periods=40,freq='h')
        frame=self.x.assign(Timestamp=times,**{'Target Timestamp':times+pd.Timedelta(hours=24)})
        source=self.root/'source.csv'; frame.to_csv(source,index=False)
        save_demand_bundle(self.model,self.bundle,list(self.x),{'train':frame.iloc[:20],'validation':frame.iloc[20:30],'test':frame.iloc[30:]},source,[],self.root,{})

    def test_roundtrip_without_training_and_reordering(self):
        with patch.object(XGBRegressor,'fit',side_effect=AssertionError('No training')):
            loaded=load_demand_bundle(self.bundle)
            np.testing.assert_allclose(loaded.predict(self.x),self.model.predict(self.x),rtol=0,atol=1e-6)
            shuffled=self.x[['hour','lag']].assign(unused='ignored')
            np.testing.assert_array_equal(loaded.predict(shuffled),loaded.predict(self.x))
            self.assertEqual(loaded.metadata['features'][0]['dtype'],'float64')

    def test_missing_duplicate_and_invalid_features(self):
        loaded=load_demand_bundle(self.bundle)
        bad_inputs=[self.x.drop(columns='lag'),self.x.assign(lag='1'),self.x.assign(lag=np.inf),self.x.assign(lag=np.nan),self.x.assign(lag=True),self.x.assign(lag=1+2j),self.x.iloc[:0],pd.concat([self.x,self.x[['lag']]],axis=1)]
        for frame in bad_inputs:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError): loaded.predict(frame)

    def test_corrupt_metadata(self):
        path=self.bundle/'metadata.json'
        original=path.read_text()
        for key,value in [('best_iteration',999),('forecast_horizon_hours',1),('parameters',{'objective':'reg:absoluteerror'}),('splits',{}),('features',[{'name':'wrong','dtype':'float64'}])]:
            metadata=json.loads(original); metadata[key]=value; path.write_text(json.dumps(metadata))
            with self.subTest(key=key):
                with self.assertRaises(ValueError): load_demand_bundle(self.bundle)
        path.write_text('{invalid JSON')
        with self.assertRaises(ValueError): load_demand_bundle(self.bundle)

    def test_corrupt_model(self):
        path=self.bundle/'model.json'; path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'hash'): load_demand_bundle(self.bundle)


if __name__=='__main__': unittest.main()
