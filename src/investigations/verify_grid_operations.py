"""Independently reconstruct exported forecast scores, dispatch and costs."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.curve_features import COMPONENTS,CurveHistory
from common.curve_model import GridCurveForecaster,digest,json_write
from common.battery_dispatch import Scenario

ROOT=Path(__file__).resolve().parents[2]
KEYS=['scenario','policy','origin']


def close(actual,expected,atol=2e-5):
    np.testing.assert_allclose(np.asarray(actual,dtype=float),np.asarray(expected,dtype=float),atol=atol,rtol=1e-11,equal_nan=True)


def trusted_targets(source,targets):
    values=source[[c+'_mw' for c in COMPONENTS]]
    bad=~np.isfinite(values).all(axis=1) | values.lt(0).any(axis=1) | source.demand_mw.le(0)
    bad |= values.eq(0).all(axis=1) | (source.index==pd.Timestamp('2025-04-04 16:00'))
    contiguous=source.index.to_series().diff().eq(pd.Timedelta(hours=1))
    for c in COMPONENTS:
        value=source[c+'_mw']
        groups=(~(value.eq(value.shift()) & contiguous)).cumsum()
        frozen=groups.map(groups.value_counts()).ge(6) & np.isfinite(value)
        if c=='solar':frozen &= value.gt(500)
        bad |= frozen
    if 'solar_wind_mw' in source:bad |= (source.solar_wind_mw-source.wind_mw-source.solar_mw).abs().gt(1)
    return (~bad).reindex(pd.DatetimeIndex(targets)).fillna(False).to_numpy(dtype=bool)


def check_forecast_metrics(rows,metrics):
    groups={}
    for h,group in rows.groupby('horizon',sort=False):
        groups[(h,'all')]=group
        for month,part in group.groupby(group.target_timestamp.dt.to_period('M').astype(str),sort=False):
            groups[(h,month)]=part
    for _,r in metrics.iterrows():
        g=groups[(r.horizon,r.month)]
        assert int(r.nominal_rows)==len(g)
        assert int(r.trusted_targets)==int(g.trusted_target.sum())
        g=g[g['score_'+r.cohort]]
        assert int(r.rows)==len(g)
        expected_coverage=100*len(g)/r.trusted_targets if r.trusted_targets else np.nan
        close(r.coverage_trusted_pct,expected_coverage)
        actual=g.actual_net_mw.to_numpy() if r.quantity=='net' else g['actual_'+r.quantity+'_mw'].to_numpy()
        predictions=[g.net_demand_mw.to_numpy(),g.baseline_net_demand_mw.to_numpy()] if r.quantity=='net' else [g[r.quantity+'_mw'].to_numpy(),g['baseline_'+r.quantity+'_mw'].to_numpy()]
        maes=[]
        for prefix,pred in zip(['forecast','baseline'],predictions):
            delta=pred-actual
            scores=[float(np.abs(delta).mean()),float(np.sqrt(np.mean(delta**2))),
                float(100*np.abs(delta).sum()/np.abs(actual).sum()) if np.abs(actual).sum() else np.nan,
                float(delta.mean())] if len(actual) else [np.nan]*4
            for name,value in zip(['mae_mw','rmse_mw','wape_pct','mean_signed_error_mw'],scores):close(r[prefix+'_'+name],value)
            maes.append(scores[0])
        close(r.mae_improvement_mw,maes[1]-maes[0])
        close(r.mae_improvement_pct,100*(maes[1]-maes[0])/maes[1] if maes[1]>0 else np.nan)


def check_physics(g,s,planned=False):
    tol=2e-5
    assert (g[['grid_mw','charge_mw','discharge_mw','renewable_used_mw','unmet_mw']].to_numpy()>=-tol).all()
    assert (g.grid_mw<=s.grid_limit_mw+tol).all()
    assert (g[['charge_mw','discharge_mw']].to_numpy()<=s.power+tol).all()
    assert (np.minimum(g.charge_mw,g.discharge_mw)<=tol).all()
    assert (g.renewable_used_mw<=g.wind_mw+g.solar_mw+tol).all()
    assert (g.unmet_mw<=g.demand_mw+tol).all()
    assert (g[['soc_start_mwh','soc_end_mwh']].to_numpy()>=s.minimum_soc_mwh-tol).all()
    assert (g[['soc_start_mwh','soc_end_mwh']].to_numpy()<=s.maximum_soc_mwh+tol).all()
    close(g.grid_mw+g.renewable_used_mw+g.discharge_mw+g.unmet_mw,g.demand_mw+g.charge_mw)
    close(g.soc_end_mwh,g.soc_start_mwh+s.charge_efficiency*g.charge_mw-g.discharge_mw/s.discharge_efficiency)
    close(g.curtailment_mw,g.wind_mw+g.solar_mw-g.renewable_used_mw)
    prices=np.asarray(s.hourly_prices_inr_per_mwh)[g.target_timestamp.dt.hour]
    close(g.price_inr_per_mwh,prices)
    close(g.electricity_cost_inr,g.grid_mw*prices)
    close(g.wear_cost_inr,(g.charge_mw+g.discharge_mw)*s.wear_cost_inr_per_mwh)


def check_operation_metrics(rows,metrics):
    groups={}
    for (scenario,policy),group in rows.groupby(['scenario','policy'],sort=False):
        groups[(scenario,policy,'all')]=group
        for month,part in group.groupby(group.target_timestamp.dt.to_period('M').astype(str),sort=False):
            groups[(scenario,policy,month)]=part
    for _,r in metrics.iterrows():
        g=groups[(r.scenario,r.policy,r.month)]
        assert len(g)==int(r.nominal_hours)
        t=g[g.trusted_target]
        assert len(t)==int(r.trusted_hours)
        assert len(g)-len(t)==int(r.excluded_hours)
        close(r.realized_electricity_cost_inr,t.electricity_cost_inr.sum(),.02)
        close(r.realized_wear_cost_inr,t.wear_cost_inr.sum(),.02)
        close(r.realized_total_cost_inr,(t.electricity_cost_inr+t.wear_cost_inr).sum(),.02)
        close(r.demand_mwh,t.demand_mw.sum())
        close(r.unserved_mwh,t.unmet_mw.sum())
        assert int(r.unserved_hours)==int(t.unmet_mw.gt(2e-5).sum())
        close(r.demand_served_pct,100*(1-t.unmet_mw.sum()/t.demand_mw.sum()) if t.demand_mw.sum() else np.nan)
        close(r.peak_grid_import_mw,t.grid_mw.max())
        close(r.curtailment_mwh,t.curtailment_mw.sum())
        close(r.battery_throughput_mwh,(t.charge_mw+t.discharge_mw).sum())
        assert int(r.episode_starts)==int(t.episode_start.sum())
        assert int(r.unavailable_plan_hours)==int(t.plan_status.eq('unavailable').sum())
        assert int(r.execution_adjustment_hours)==int(t.action_adjusted.sum())
        close(r.available_plan_pct,100*t.plan_status.eq('optimal').mean() if r.policy!='no_battery' and len(t) else np.nan)


def check_cli(root,raw):
    from forecasting.predict_grid_curve import main
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);output=tmp/'curve.json'
        argv=['predict_grid_curve','--origin',str(raw.timestamp.max()),'--history',str(root/'data/processed/full_hourly_stacked.csv'),
              '--root',str(root),'--scenario','operations_v1','--soc-mwh','40','--output',str(output)]
        stdout=io.StringIO()
        with patch.object(sys,'argv',argv),redirect_stdout(stdout):status=main()
        assert status==0
        result=json.loads(stdout.getvalue())
        assert result==json.loads(output.read_text())
        assert len(result['points'])==24 and result['simulation']['plan']['status']=='optimal'
        assert all(pd.Timestamp(p['target_timestamp'])>raw.timestamp.max() for p in result['points'])
        old=digest(output)
        with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):assert main()==1
        assert digest(output)==old
        blocked_history=raw.tail(169).drop(columns=['solar_wind_mw'],errors='ignore').copy()
        blocked_history['wind_mw']=-1.
        blocked_path=tmp/'blocked.csv';blocked_history.to_csv(blocked_path,index=False)
        args=['predict_grid_curve','--origin',str(raw.timestamp.max()),'--history',str(blocked_path),
              '--root',str(root),'--scenario','operations_v1','--soc-mwh','40']
        stdout=io.StringIO()
        with patch.object(sys,'argv',args),redirect_stdout(stdout):assert main()==2
        blocked=json.loads(stdout.getvalue())
        assert not blocked['complete_curve'] and blocked['simulation']['plan']['status']=='unavailable'
        assert all(p['components']['wind']['status']=='blocked' for p in blocked['points'])
        assert all(p['components']['demand']['status']=='ok' for p in blocked['points'])
        args=['predict_grid_curve','--origin','2025-01-09 15:00:00','--history',str(root/'data/processed/full_hourly_stacked.csv'),
              '--root',str(root)]
        stdout=io.StringIO()
        with patch.object(sys,'argv',args),redirect_stdout(stdout):assert main()==0
        fallback=json.loads(stdout.getvalue())
        assert fallback['complete_curve']
        assert any(p['components']['demand']['status']=='fallback' for p in fallback['points'])
    return {'load_only_cli':True,'exactly_24_unlabelled_future_targets':True,
            'json_stdout_equals_output':True,'existing_output_protected':True,
            'blocked_component_isolation_and_no_plan':True,'labelled_historical_fallback':True}


def verify(root):
    directory=root/'reports/experiments/grid_operations_v1'
    summary=json.loads((directory/'summary.json').read_text())
    service=GridCurveForecaster(root,strict=True)
    assert digest(service.directory/'metadata.json')==summary['model_manifest_sha256']
    assert digest(root/summary['source']['path'])==summary['source']['sha256']
    assert digest(root/'configs/operations_v1.json')==summary['scenario_sha256']
    for p,expected in summary['code_sha256'].items():assert digest(root/p)==expected,f'Changed code: {p}'
    for key,sha in summary['model_sha256'].items():assert service.manifest['entries'][key]['model_sha256']==sha
    raw=pd.read_csv(root/summary['source']['path'],parse_dates=['timestamp'])
    source=raw.set_index('timestamp')
    rows=pd.read_csv(directory/'forecast_rows.csv.gz',parse_dates=['origin','target_timestamp'],float_precision='round_trip')
    assert not rows.duplicated(['origin','horizon']).any()
    assert (rows.groupby('origin').horizon.agg(list).map(lambda x:x==list(range(1,25)))).all()
    assert (rows.target_timestamp-rows.origin).eq(pd.to_timedelta(rows.horizon,unit='h')).all()
    assert rows.origin.nunique()==summary['nominal_origins']
    assert len(rows)==summary['forecast_exposures']
    assert np.array_equal(rows.trusted_target,trusted_targets(source,rows.target_timestamp))
    for c in COMPONENTS:close(rows['actual_'+c+'_mw'],source[c+'_mw'].reindex(rows.target_timestamp).to_numpy())
    close(rows.actual_net_mw,rows.actual_demand_mw-rows.actual_wind_mw-rows.actual_solar_mw)
    close(rows.net_demand_mw,rows.demand_mw-rows.wind_mw-rows.solar_mw)
    close(rows.baseline_net_demand_mw,rows.baseline_demand_mw-rows.baseline_wind_mw-rows.baseline_solar_mw)
    for cohort in ['selected_only','delivered']:
        available=rows[[c+'_status' for c in COMPONENTS]].eq('ok').all(axis=1) if cohort=='selected_only' else ~rows[[c+'_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
        base=~rows[['baseline_'+c+'_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
        assert np.array_equal(rows['score_'+cohort],rows.trusted_target & available & base)
    context=CurveHistory(raw)
    origins=pd.DatetimeIndex(rows.origin.drop_duplicates())
    for baseline in [False,True]:
        fresh=service.forecast_rows(context,origins,baseline_only=baseline)
        prefix='baseline_' if baseline else ''
        for c in COMPONENTS:
            close(rows[prefix+c+'_mw'],fresh[c+'_mw'],1e-6)
            for field in ['method','status','reasons','unavailable_checks']:
                assert np.array_equal(rows[prefix+c+'_'+field].fillna('').astype(str).to_numpy(),
                                      fresh[c+'_'+field].fillna('').astype(str).to_numpy())
    check_forecast_metrics(rows,pd.read_csv(directory/'forecast_metrics.csv'))
    dispatch=pd.read_csv(directory/'dispatch_rows.csv.gz',parse_dates=['origin','target_timestamp'],float_precision='round_trip')
    assert not dispatch.duplicated(KEYS).any()
    assert (dispatch.target_timestamp-dispatch.origin).eq(pd.Timedelta(hours=1)).all()
    assert np.array_equal(dispatch.trusted_target,trusted_targets(source,dispatch.target_timestamp))
    scenarios={s['scenario_id']:Scenario(**s) for s in summary['scenarios']}
    expected_initial={}
    for (name,policy),g in dispatch.groupby(['scenario','policy']):
        assert len(g)==len(origins)
        s=scenarios[name]
        if policy=='no_battery':s=replace(s,battery_capacity_mwh=0.)
        g=g.sort_values('origin')
        previous=g.soc_end_mwh.shift()
        expected=previous.where(g.trusted_target.shift(fill_value=False),s.initial_soc_mwh).fillna(s.initial_soc_mwh)
        for origin,state in zip(g.origin,expected):expected_initial[(name,policy,origin)]=state
        t=g[g.trusted_target]
        check_physics(t,s)
        close(t.soc_start_mwh,expected.loc[t.index])
        for c in COMPONENTS:close(t[c+'_mw'],source[c+'_mw'].reindex(t.target_timestamp).to_numpy()*s.scale_factor)
        charge=np.minimum.reduce([np.maximum(t.planned_charge_mw.to_numpy(),0),
            np.full(len(t),s.power),np.maximum(0,(s.maximum_soc_mwh-t.soc_start_mwh)/s.charge_efficiency),
            np.maximum(0,t.wind_mw+t.solar_mw+s.grid_limit_mw-t.demand_mw)])
        discharge=np.minimum.reduce([np.maximum(t.planned_discharge_mw.to_numpy(),0),
            np.full(len(t),s.power),np.maximum(0,(t.soc_start_mwh-s.minimum_soc_mwh)*s.discharge_efficiency),
            np.maximum(0,t.demand_mw-t.wind_mw-t.solar_mw)])
        close(t.charge_mw,charge);close(t.discharge_mw,discharge)
        unavailable=t.plan_status.eq('unavailable')
        close(t.loc[unavailable,['charge_mw','discharge_mw']],0)
        starts=t.episode_start
        close(t.loc[starts,'soc_start_mwh'],s.initial_soc_mwh)
    check_operation_metrics(dispatch,pd.read_csv(directory/'operations_metrics.csv'))
    forecasts=rows.set_index(['origin','horizon'])
    dispatch_index=dispatch.set_index(KEYS)
    plan_count=0;plan_rows=0
    for chunk in pd.read_csv(directory/'plans.csv.gz',chunksize=48000,parse_dates=['origin','target_timestamp'],float_precision='round_trip'):
        assert (chunk.target_timestamp-chunk.origin).eq(pd.to_timedelta(chunk.horizon,unit='h')).all()
        for name,g in chunk.groupby('scenario'):
            s=scenarios[name];check_physics(g,s,planned=True)
            reference=forecasts.reindex(pd.MultiIndex.from_frame(g[['origin','horizon']]))
            for c in COMPONENTS:
                values=np.where(g.policy.eq('baseline_battery'),reference['baseline_'+c+'_mw'],reference[c+'_mw'])*s.scale_factor
                close(g[c+'_mw'],values)
        for key,g in chunk.groupby(KEYS,sort=False):
            assert g.horizon.tolist()==list(range(1,25))
            s=scenarios[key[0]]
            close(g.soc_start_mwh.iloc[0],expected_initial[key])
            close(g.soc_start_mwh.iloc[1:].to_numpy(),g.soc_end_mwh.iloc[:-1].to_numpy())
            assert g.soc_end_mwh.iloc[-1]>=s.terminal_soc_mwh-2e-5
            assert g.unmet_mw.sum()<=g.minimum_unmet_mwh.iloc[0]+2e-5
            r=dispatch_index.loc[key]
            close(r.planned_charge_mw,g.charge_mw.iloc[0]);close(r.planned_discharge_mw,g.discharge_mw.iloc[0])
            close(r.plan_predicted_unmet_mwh,g.unmet_mw.sum())
            close(r.plan_predicted_electricity_cost_inr,g.electricity_cost_inr.sum(),.02)
            close(r.plan_predicted_wear_cost_inr,g.wear_cost_inr.sum(),.02)
            assert r.plan_status=='optimal'
            plan_count+=1
        plan_rows+=len(chunk)
    assert plan_rows==summary['plan_rows']
    assert plan_count==int(dispatch.plan_status.eq('optimal').sum())
    result={'status':'passed','training_disabled':True,'forecast_exposures_verified':len(rows),
        'origins':len(origins),'dispatch_rows_verified':len(dispatch),'plans_verified':plan_count,'plan_rows_verified':plan_rows,
        'checks':['independent actual/target-policy alignment','fresh load-only selected/baseline replay',
            'all horizon/month forecast metrics','identical paired scoring masks','all scenario/month realized costs and service',
            'plan energy balance, efficiency, bounds and terminal reserve','carried executed SOC and episode resets',
            'first-action recourse and unavailable-plan idle behavior','model/source/code/scenario hashes'],
        'cli':check_cli(root,raw),'verification_code_sha256':digest(Path(__file__)),
        'normalization_code_sha256':digest(root/'src/common/inference_history.py'),
        'output_sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='verification.json'}}
    return result


def disabled(*args,**kwargs):raise AssertionError('Training disabled during verification')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    with patch.object(xgb.XGBRegressor,'fit',disabled),patch.object(xgb,'train',disabled),patch.object(xgb.Booster,'update',disabled):
        report=verify(args.root)
    json_write(args.root/'reports/experiments/grid_operations_v1/verification.json',report)
    print(json.dumps(report,indent=2,allow_nan=False))
