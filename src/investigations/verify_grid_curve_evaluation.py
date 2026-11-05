"""Verify forecast-only exports with training and battery solving disabled."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.curve_features import COMPONENTS,CurveHistory
from common.curve_model import GridCurveForecaster,digest,json_write
from investigations.verify_grid_operations import close,trusted_targets,check_forecast_metrics

ROOT=Path(__file__).resolve().parents[2]


def forbidden(*args,**kwargs):raise AssertionError('Training and battery work are disabled in forecast-only verification')


def verify(root):
    directory=root/'reports/experiments/grid_curve_v1'
    summary=json.loads((directory/'summary.json').read_text())
    service=GridCurveForecaster(root,strict=True)
    assert digest(service.directory/'metadata.json')==summary['model_manifest_sha256']
    native=json.loads((directory/'verification.json').read_text())
    assert native['status']=='passed' and native['manifest_sha256']==summary['model_manifest_sha256']
    assert digest(root/summary['source']['path'])==summary['source']['sha256']
    for path,expected in summary['code_sha256'].items():assert digest(root/path)==expected,f'Changed code: {path}'
    for key,expected in summary['model_sha256'].items():assert service.manifest['entries'][key]['model_sha256']==expected
    raw=pd.read_csv(root/summary['source']['path'],parse_dates=['timestamp'])
    source=raw.set_index('timestamp')
    rows=pd.read_csv(directory/'forecast_rows.csv.gz',parse_dates=['origin','target_timestamp'],float_precision='round_trip')
    assert not rows.duplicated(['origin','horizon']).any()
    assert rows.groupby('origin').horizon.agg(list).map(lambda x:x==list(range(1,25))).all()
    assert (rows.target_timestamp-rows.origin).eq(pd.to_timedelta(rows.horizon,unit='h')).all()
    assert len(rows)==summary['forecast_exposures'] and rows.origin.nunique()==summary['nominal_origins']
    assert np.array_equal(rows.trusted_target,trusted_targets(source,rows.target_timestamp))
    for c in COMPONENTS:close(rows['actual_'+c+'_mw'],source[c+'_mw'].reindex(rows.target_timestamp).to_numpy())
    close(rows.actual_net_mw,rows.actual_demand_mw-rows.actual_wind_mw-rows.actual_solar_mw)
    close(rows.net_demand_mw,rows.demand_mw-rows.wind_mw-rows.solar_mw)
    close(rows.baseline_net_demand_mw,rows.baseline_demand_mw-rows.baseline_wind_mw-rows.baseline_solar_mw)
    base_available=~rows[['baseline_'+c+'_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
    for cohort in ['selected_only','delivered']:
        status=rows[[c+'_status' for c in COMPONENTS]]
        available=status.eq('ok').all(axis=1) if cohort=='selected_only' else ~status.eq('blocked').any(axis=1)
        assert np.array_equal(rows['score_'+cohort],rows.trusted_target & available & base_available)
    origins=pd.DatetimeIndex(rows.origin.drop_duplicates())
    context=CurveHistory(raw)
    for baseline in [False,True]:
        fresh=service.forecast_rows(context,origins,baseline_only=baseline)
        prefix='baseline_' if baseline else ''
        for c in COMPONENTS:
            close(rows[prefix+c+'_mw'],fresh[c+'_mw'],1e-6)
            for field in ['method','status','reasons','unavailable_checks']:
                assert np.array_equal(rows[prefix+c+'_'+field].fillna('').astype(str).to_numpy(),fresh[c+'_'+field].fillna('').astype(str).to_numpy())
    metrics=pd.read_csv(directory/'forecast_metrics.csv')
    check_forecast_metrics(rows,metrics)
    from forecasting.predict_grid_curve import main
    stdout=io.StringIO()
    argv=['predict_grid_curve','--origin',str(raw.timestamp.max()),'--history',str(root/summary['source']['path']),'--root',str(root)]
    with patch.object(sys,'argv',argv),redirect_stdout(stdout):assert main()==0
    output=json.loads(stdout.getvalue())
    assert len(output['points'])==24 and 'simulation' not in output
    assert all(pd.Timestamp(p['target_timestamp'])>raw.timestamp.max() for p in output['points'])
    return {'status':'passed','training_disabled':True,'battery_work_performed':False,
        'origins':len(origins),'forecast_exposures_verified':len(rows),'metric_rows_independently_verified':len(metrics),
        'native_model_reference_rows_previously_verified':native['aligned_reference_rows'],
        'checks':['one-to-one origin/horizon/target alignment','actuals against immutable source',
            'independent retrospective target mask','identical candidate/baseline scoring masks',
            'all delivered forecasts/statuses/methods reproduced load-only',
            'all component/net MAE, RMSE, WAPE and signed errors by horizon/month',
            'forecast-only CLI emits 24 future targets without target labels','source/model/code hashes'],
        'verification_code_sha256':digest(Path(__file__)),
        'independent_metric_checker_sha256':digest(root/'src/investigations/verify_grid_operations.py'),
        'output_sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='evaluation_verification.json'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    with patch.object(xgb.XGBRegressor,'fit',forbidden),patch.object(xgb,'train',forbidden),patch.object(xgb.Booster,'update',forbidden),patch('common.battery_dispatch.milp',forbidden):
        result=verify(args.root)
    json_write(args.root/'reports/experiments/grid_curve_v1/evaluation_verification.json',result)
    print(json.dumps(result,indent=2,allow_nan=False))
