"""Integration verification only: compare history-only inference to saved experiments.

The production API never imports training code. This verifier imports historical
feature builders solely as reference implementations; all training is disabled.
"""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid_prediction import GridForecaster, ROOT
from common.inference_history import prepare_history, demand_features, wind_features, DEMAND_COLUMNS, WIND_COLUMNS
from common.features import build_features
from forecasting.renewable_v3_full_history import build_frame
from forecasting.predict_grid import main as cli_main


def verify(root):
    raw = pd.read_csv(root/'data/processed/full_hourly_stacked.csv')
    engine = GridForecaster(root)
    past = prepare_history(raw,pd.to_datetime(raw.timestamp).max())
    demand_saved = pd.read_csv(root/'reports/experiments/xgboost_demand_predictions.csv')
    wind_saved = pd.read_csv(root/'reports/experiments/renewable_v3_wind_predictions.csv')
    source = pd.read_csv(root/'data/processed/demand_model_data.csv',parse_dates=['Timestamp'])
    demand_reference = build_features(source).set_index('Timestamp')
    wind_raw = raw.copy(); wind_raw.timestamp = pd.to_datetime(wind_raw.timestamp)
    bad = wind_raw[['demand_mw','wind_mw','solar_mw','total_gen_mw']].eq(0).all(axis=1)
    bad |= (wind_raw.solar_wind_mw-wind_raw.wind_mw-wind_raw.solar_mw).abs().gt(1)
    wind_reference, _ = build_frame(wind_raw,'wind_mw',bad)
    wind_reference = wind_reference.set_index('origin')
    demand_rows, wind_rows = [], []
    max_demand_feature_error = 0.
    max_wind_feature_error = 0.
    for stamp in pd.to_datetime(demand_saved['Forecast Origin']):
        features = demand_features(past,stamp)
        reference = demand_reference.loc[stamp, DEMAND_COLUMNS].to_numpy(dtype=float)
        error = float(np.max(np.abs(features.to_numpy(dtype=float)[0]-reference)))
        max_demand_feature_error=max(max_demand_feature_error,error)
        np.testing.assert_allclose(features.to_numpy(dtype=float)[0],reference,rtol=0,atol=1e-8)
        demand_rows.append(features)
    for stamp in pd.to_datetime(wind_saved.origin):
        features = wind_features(past,stamp)
        reference = wind_reference.loc[stamp,WIND_COLUMNS].to_numpy(dtype=float)
        error = float(np.max(np.abs(features.to_numpy(dtype=float)[0]-reference)))
        max_wind_feature_error=max(max_wind_feature_error,error)
        np.testing.assert_allclose(features.to_numpy(dtype=float)[0],reference,rtol=0,atol=1e-8)
        wind_rows.append(features)
    demand = engine.models['demand'].predict(pd.concat(demand_rows,ignore_index=True))
    wind_input = pd.concat(wind_rows,ignore_index=True)
    wind = wind_input.prior_day.to_numpy() + engine.models['wind'].predict(wind_input,iteration_range=(0,82))
    demand_delta = float(np.max(np.abs(demand-demand_saved['XGBoost Forecast (MW)'])))
    wind_delta = float(np.max(np.abs(wind-wind_saved.selected)))
    np.testing.assert_allclose(demand,demand_saved['XGBoost Forecast (MW)'],rtol=0,atol=.02)
    np.testing.assert_allclose(wind,wind_saved.selected,rtol=0,atol=.02)
    # Exercise full quality-aware orchestration at multiple seasonal origins.
    indices=np.unique(np.linspace(0,len(demand_saved)-1,32,dtype=int))
    normal_count=0
    for i in indices:
        stamp=pd.Timestamp(demand_saved['Forecast Origin'].iloc[i])
        report=engine.forecast(raw,stamp)
        components=report['components']
        if components['demand']['status']=='ok':
            np.testing.assert_allclose(components['demand']['forecast_mw'],demand_saved['XGBoost Forecast (MW)'].iloc[i],rtol=0,atol=.02)
            normal_count+=1
        if components['wind']['status']=='ok':
            matching=wind_saved.loc[wind_saved.origin.eq(str(stamp))]
            if len(matching)==1:
                np.testing.assert_allclose(components['wind']['forecast_mw'],matching.selected.iloc[0],rtol=0,atol=.02)
        self_solar=float(past.at[stamp,'solar_mw'])
        if components['solar']['status']=='ok':
            assert components['solar']['forecast_mw']==self_solar
        assert all(c['target_timestamp']==str(stamp+pd.Timedelta(hours=24)) for c in components.values())
        cutoff=raw.loc[pd.to_datetime(raw.timestamp)<=stamp].copy()
        assert report==engine.forecast(cutoff,stamp)
        future=raw.copy(); mask=pd.to_datetime(future.timestamp)>stamp
        future.loc[mask,['demand_mw','wind_mw','solar_mw','solar_wind_mw']]=np.nan
        assert report==engine.forecast(future,stamp)
    examples={
        'normal':engine.forecast(raw,'2025-06-24 23:00:00'),
        'labelled_fallback':engine.forecast(raw,'2025-01-09 15:00:00'),
        'blocked_demand_component':engine.forecast(raw,'2025-01-09 02:00:00'),
    }
    assert examples['normal']['status']=='ok'
    assert examples['labelled_fallback']['components']['demand']['status']=='fallback'
    assert examples['blocked_demand_component']['components']['demand']['status']=='blocked'
    # Existing historical policy still excludes the January target.
    jan=demand_reference.loc[pd.Timestamp('2025-01-09 15:00:00')]
    assert not jan.quality_eligible
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory)
        normal_file=directory/'normal.json'
        stdout=io.StringIO()
        with patch.object(sys,'argv',['predict_grid','--origin','2025-06-24 23:00:00','--history',str(root/'data/processed/full_hourly_stacked.csv'),'--root',str(root),'--output',str(normal_file)]),redirect_stdout(stdout):
            code=cli_main()
        assert code==0 and stdout.getvalue()==normal_file.read_text()
        assert json.loads(stdout.getvalue())==examples['normal']
        invalid=io.StringIO()
        with patch.object(sys,'argv',['predict_grid','--origin','2025-06-24 23:01:00','--history',str(root/'data/processed/full_hourly_stacked.csv'),'--root',str(root)]),redirect_stderr(invalid):
            assert cli_main()==1
        assert json.loads(invalid.getvalue())['status']=='error'
        zeros=raw.copy(); zeros.loc[zeros.index[-1],['demand_mw','wind_mw','solar_mw','solar_wind_mw']]=0
        zeros_path=directory/'zero.csv'; zeros.to_csv(zeros_path,index=False)
        with patch.object(sys,'argv',['predict_grid','--origin','2025-06-24 23:00:00','--history',str(zeros_path),'--root',str(root)]),redirect_stdout(io.StringIO()):
            assert cli_main()==2
    return {
        'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_disabled':True,'all_origins_use_common_24h_target':True,
        'demand':{'saved_test_rows_compared':len(demand_saved),'max_feature_difference':max_demand_feature_error,'max_prediction_difference_mw':demand_delta},
        'wind':{'saved_test_rows_compared':len(wind_saved),'max_feature_difference':max_wind_feature_error,'max_prediction_difference_mw':wind_delta},
        'quality_orchestration':{'seasonal_origins_checked':len(indices),'demand_selected_model_origins':normal_count,
                                 'truncation_and_future_measurement_invariance':True},
        'cli':{'json_stdout_equals_optional_output':True,'invalid_request_exit_code':1,'all_blocked_exit_code':2},
        'january_target_excluded_from_historical_scoring':True,
        'note':'Numerical parity uses all saved rows. The stricter inference policy may instead return labelled persistence or block on some historically scored rows; those outputs are not represented as new benchmark scores.',
        'examples':examples,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with ExitStack() as stack:
        for obj,name in [(xgboost.XGBRegressor,'fit'),(xgboost,'train'),(xgboost.Booster,'update')]:
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden during prediction verification')))
        report=verify(args.root)
    rendered=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__=='__main__': main()
