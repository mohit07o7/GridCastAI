"""Independent-process curve feature and native-model reproduction checks."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.curve_features import COMPONENTS, FEATURES, HOUR, CurveHistory
from common.curve_model import GridCurveForecaster, digest, predict_candidate, json_write

ROOT=Path(__file__).resolve().parents[2]


def forbidden(*args,**kwargs):
    raise AssertionError('Training is disabled during curve verification')


def independent_features(raw,origin,horizon,component):
    """Scalar definitions independently reconstructed from the prefix."""
    v=raw.loc[:origin,f'{component}_mw']
    values={f'lag_{lag}':v.at[origin-lag*HOUR] for lag in [0,1,3,6,24,48,168]}
    for w in [24,168]:
        last=v.loc[origin-(w-1)*HOUR:origin]
        values[f'recent_mean_{w}']=float(np.mean(last.to_numpy()))
        values[f'recent_std_{w}']=float(np.std(last.to_numpy(),ddof=1))
    values['target_previous_day']=v.at[origin+(horizon-24)*HOUR]
    values['target_previous_week']=v.at[origin+(horizon-168)*HOUR]
    target=origin+horizon*HOUR
    for name,value,period in [('hour',target.hour,24),('dow',target.dayofweek,7),('month',target.month-1,12)]:
        values[f'target_{name}_sin']=np.sin(2*np.pi*value/period)
        values[f'target_{name}_cos']=np.cos(2*np.pi*value/period)
    values['target_year']=float(target.year)
    return np.array([values[f] for f in FEATURES])


def verify(root):
    service=GridCurveForecaster(root,strict=True)
    meta=service.manifest
    for name,expected in meta['code_sha256'].items():
        assert digest(root/name)==expected, f'Changed training/feature code: {name}'
    assert digest(root/meta['source']['path'])==meta['source']['sha256']
    path=service.directory/meta['test_reference']['path']
    assert digest(path)==meta['test_reference']['sha256']
    reference=pd.read_csv(path,parse_dates=['origin','target_timestamp'],float_precision='round_trip')
    assert len(reference)==meta['test_reference']['rows']
    assert not reference.duplicated(['origin','component','horizon']).any()
    raw=pd.read_csv(root/meta['source']['path'],parse_dates=['timestamp'])
    context=CurveHistory(raw)
    indexed=raw.set_index('timestamp')
    max_prediction_error=0.
    for (c,h),group in reference.groupby(['component','horizon']):
        h=int(h)
        origins=pd.DatetimeIndex(group.origin)
        assert (group.target_timestamp-group.origin).eq(h*HOUR).all()
        expected=indexed[f'{c}_mw'].reindex(group.target_timestamp).to_numpy()
        np.testing.assert_allclose(group.actual,expected,atol=1e-8,rtol=0)
        key=f'{c}_h{h:02d}'
        prediction=predict_candidate(service.models[key],context.features(c,h,origins),meta['entries'][key])
        max_prediction_error=max(max_prediction_error,float(np.max(np.abs(prediction-group.candidate_mw))))
        np.testing.assert_allclose(prediction,group.candidate_mw,atol=1e-6,rtol=0)
    origins=pd.DatetimeIndex(reference.origin.drop_duplicates().sort_values())
    samples=origins[np.unique(np.linspace(0,len(origins)-1,24,dtype=int))]
    feature_error=0.
    for origin in samples:
        prefix=CurveHistory(raw,origin)
        for c in COMPONENTS:
            for h in range(1,25):
                got=context.features(c,h,[origin]).iloc[0].to_numpy()
                expected=independent_features(indexed,origin,h,c)
                feature_error=max(feature_error,float(np.max(np.abs(got-expected))))
                np.testing.assert_allclose(got,expected,atol=1e-5,rtol=1e-10)
            for offset in [None,0,7,23,144,167]:
                pd.testing.assert_frame_equal(context.quality(c,[origin],offset),prefix.quality(c,[origin],offset))
        pd.testing.assert_frame_equal(service.forecast_rows(context,[origin]),service.forecast_rows(prefix,[origin]))
    origin=samples[len(samples)//2]
    normal=service.forecast(raw,origin)
    modified=raw.copy()
    modified.loc[modified.timestamp>origin,['demand_mw','wind_mw','solar_mw']]=np.nan
    assert normal==service.forecast(modified,origin)
    assert normal==service.forecast(raw[raw.timestamp<=origin],origin)
    assert len(normal['points'])==24
    assert not any('train_grid_curve' in name for name in sys.modules)
    return {'status':'passed','training_disabled':True,'candidates':72,
        'aligned_reference_rows':len(reference),'unique_origins':len(origins),
        'max_loaded_vs_in_memory_error_mw':max_prediction_error,
        'prediction_tolerance_mw':1e-6,'independent_feature_origins':len(samples),
        'max_independent_feature_absolute_error':feature_error,
        'future_data_invariance':True,'prefix_batch_quality_parity':True,
        'manifest_sha256':digest(service.directory/'metadata.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with patch.object(xgb.XGBRegressor,'fit',forbidden),patch.object(xgb,'train',forbidden),patch.object(xgb.Booster,'update',forbidden):
        report=verify(args.root)
    if args.output:json_write(args.output,report)
    print(json.dumps(report,indent=2,allow_nan=False))
