"""Independently recompute exported grid-evaluation results (no prediction/training)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import platform
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
KEYS=['origin','target_timestamp']
COMPONENTS=['demand','wind','solar']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual,expected):
    if expected is None:
        assert actual is None or pd.isna(actual), (actual,expected)
    else:
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8,equal_nan=True)


def read_rows(path):
    frame=pd.read_csv(path,parse_dates=KEYS)
    assert not frame.duplicated(KEYS).any()
    assert frame.target_timestamp.sub(frame.origin).eq(pd.Timedelta(hours=24)).all()
    return frame


def independently_trusted(source, targets):
    finite=np.isfinite(source[['demand_mw','wind_mw','solar_mw']]).all(axis=1)
    bad=~finite | source[['demand_mw','wind_mw','solar_mw']].lt(0).any(axis=1) | source.demand_mw.le(0)
    bad |= source[['demand_mw','wind_mw','solar_mw']].eq(0).all(axis=1)
    bad |= source.index==pd.Timestamp('2025-04-04 16:00:00')
    for component in COMPONENTS:
        values=source[f'{component}_mw']
        groups=values.ne(values.shift()).cumsum()
        frozen=values.groupby(groups).transform('size').ge(6) & np.isfinite(values)
        if component=='solar':frozen &= values.gt(500)
        bad |= frozen
    bad |= (source.solar_wind_mw-source.wind_mw-source.solar_mw).abs().gt(1)
    return (~bad).reindex(pd.DatetimeIndex(targets)).fillna(False).to_numpy(dtype=bool)


def check_quantities(frame,source):
    actual={c:source[f'{c}_mw'].reindex(frame.target_timestamp).to_numpy(dtype=float) for c in COMPONENTS}
    baseline={c:source[f'{c}_mw'].reindex(frame.origin).to_numpy(dtype=float) for c in COMPONENTS}
    for c in COMPONENTS:
        close(frame[f'actual_{c}'],actual[c]);close(frame[f'baseline_{c}'],baseline[c])
    for prefix in ['actual','predicted','baseline']:
        demand=frame[f'{prefix}_demand'].to_numpy(dtype=float)
        renewable=frame[f'{prefix}_wind'].to_numpy(dtype=float)+frame[f'{prefix}_solar'].to_numpy(dtype=float)
        net=demand-renewable
        share=np.divide(100*renewable,demand,out=np.full(len(demand),np.nan),where=demand>0)
        close(frame[f'{prefix}_renewable_mw'],renewable)
        close(frame[f'{prefix}_net_demand_mw'],net)
        close(frame[f'{prefix}_renewable_share_pct'],share)
        assert np.array_equal(frame[f'{prefix}_share_undefined'],~np.isfinite(share))
    for prefix in ['predicted','baseline']:
        errors={c:frame[f'{prefix}_{c}'].to_numpy(dtype=float)-actual[c] for c in COMPONENTS}
        for c in COMPONENTS:close(frame[f'{prefix}_{c}_error_mw'],errors[c])
        identity=errors['demand']-errors['wind']-errors['solar']
        close(frame[f'{prefix}_net_error_mw'],identity)
        close(frame[f'{prefix}_net_absolute_error_mw'],np.abs(identity))
        close(frame[f'{prefix}_renewable_error_mw'],errors['wind']+errors['solar'])
        close(frame[f'{prefix}_renewable_share_error_pp'],frame[f'{prefix}_renewable_share_pct']-frame.actual_renewable_share_pct)
    assert np.array_equal(frame.trusted_target,independently_trusted(source,frame.target_timestamp))


def independent_metrics(frame):
    actual=frame.actual_demand.to_numpy()-frame.actual_wind.to_numpy()-frame.actual_solar.to_numpy()
    actual_r=frame.actual_wind.to_numpy()+frame.actual_solar.to_numpy()
    result={}
    share_valid=np.isfinite(frame[[f'{p}_renewable_share_pct' for p in ['actual','predicted','baseline']]]).all(axis=1)
    result['paired_share_rows']=int(share_valid.sum())
    errors={}
    for prefix in ['predicted','baseline']:
        predicted=frame[f'{prefix}_demand'].to_numpy()-frame[f'{prefix}_wind'].to_numpy()-frame[f'{prefix}_solar'].to_numpy()
        renewable=frame[f'{prefix}_wind'].to_numpy()+frame[f'{prefix}_solar'].to_numpy()
        errors[prefix]=predicted-actual
        for label,truth,estimate in [('net',actual,predicted),('renewable',actual_r,renewable)]:
            e=estimate-truth
            result[f'{prefix}_{label}_mae_mw']=float(np.mean(np.abs(e))) if len(e) else None
            result[f'{prefix}_{label}_rmse_mw']=float(np.sqrt(np.mean(e*e))) if len(e) else None
            result[f'{prefix}_{label}_mean_signed_error_mw']=float(np.mean(e)) if len(e) else None
            denom=np.sum(np.abs(truth))
            result[f'{prefix}_{label}_wape_pct']=float(100*np.sum(np.abs(e))/denom) if len(e) and denom>0 else None
        result[f'{prefix}_renewable_share_mae_pp']=float(np.mean(np.abs(frame.loc[share_valid,f'{prefix}_renewable_share_pct']-frame.loc[share_valid,'actual_renewable_share_pct']))) if share_valid.any() else None
        result[f'{prefix}_undefined_share_rows']=int((~np.isfinite(frame[f'{prefix}_renewable_share_pct'])).sum())
    improvement=result['baseline_net_mae_mw']-result['predicted_net_mae_mw'] if len(frame) else None
    result['mae_improvement_mw']=improvement
    result['mae_improvement_pct']=100*improvement/result['baseline_net_mae_mw'] if len(frame) and result['baseline_net_mae_mw']>0 else None
    if len(frame):
        dates=frame.target_timestamp.dt.strftime('%Y-%m-%d').to_numpy()
        days,inverse=np.unique(dates,return_inverse=True)
        differences=np.abs(errors['baseline'])-np.abs(errors['predicted'])
        sums=np.bincount(inverse,weights=differences)
        counts=np.bincount(inverse)
        draws=np.random.default_rng(42).integers(0,len(days),size=(1000,len(days)))
        draws_mean=np.sum(sums[draws],axis=1)/np.sum(counts[draws],axis=1)
        bootstrap={'mae_improvement_mw':float(np.mean(differences)),
                   'ci95_low_mw':float(np.quantile(draws_mean,.025)),
                   'ci95_high_mw':float(np.quantile(draws_mean,.975)),'days':len(days),'repeats':1000,'seed':42}
    else:
        bootstrap={'mae_improvement_mw':None,'ci95_low_mw':None,'ci95_high_mw':None,'days':0,'repeats':1000,'seed':42}
    return result,bootstrap


def verify(root):
    directory=root/'reports/experiments/grid_evaluation_v1'
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['training_disabled'] is True
    for entries in ['source_artifact_sha256','code_sha256']:
        for relative,expected in summary[entries].items():
            assert digest(root/relative)==expected,relative
    source=pd.read_csv(root/'data/processed/full_hourly_stacked.csv',parse_dates=['timestamp']).set_index('timestamp')
    saved=read_rows(directory/'saved_rows.csv');replay=read_rows(directory/'replay_rows.csv');audit=read_rows(directory/'alignment_audit.csv')
    keys=list(map(tuple,pd.MultiIndex.from_frame(replay[KEYS]).tolist()))
    expected=pd.date_range(summary['nominal']['first_origin'],summary['nominal']['last_origin'],freq='h')
    assert len(replay)==len(expected)==summary['nominal']['rows']
    assert replay.origin.tolist()==expected.tolist()
    assert audit[KEYS].equals(replay[KEYS])
    assert set(pd.MultiIndex.from_frame(saved[KEYS]).tolist())==set(pd.MultiIndex.from_frame(audit.loc[audit.saved_common,KEYS]).tolist())
    for component in COMPONENTS:
        name={'demand':'xgboost_demand_predictions.csv','wind':'renewable_v3_wind_predictions.csv','solar':'renewable_v3_solar_predictions.csv'}[component]
        original=pd.read_csv(root/'reports/experiments'/name)
        if component=='demand':original=original.rename(columns={'Forecast Origin':'origin','Target Timestamp':'target_timestamp','XGBoost Forecast (MW)':'selected'})
        original['origin']=pd.to_datetime(original.origin);original['target_timestamp']=pd.to_datetime(original.target_timestamp)
        paired=saved.merge(original[KEYS+['selected']],on=KEYS,validate='one_to_one')
        assert len(paired)==len(saved)
        close(paired[f'predicted_{component}'],paired.selected)
        presence=original[KEYS].assign(expected_present=True)
        checked=audit.merge(presence,on=KEYS,how='left',validate='one_to_one')
        assert np.array_equal(checked[f'{component}_saved_present'],checked.expected_present.fillna(False))
    check_quantities(saved,source);check_quantities(replay,source)
    assert replay.inference_target_quality.eq('unknown').all()
    finite=np.isfinite(replay[[f'predicted_{c}' for c in COMPONENTS]]).all(axis=1)
    baseline_finite=np.isfinite(replay[[f'baseline_{c}' for c in COMPONENTS]]).all(axis=1)
    baseline_finite &= replay[[f'baseline_{c}' for c in COMPONENTS]].ge(0).all(axis=1)
    selected=replay[[f'{c}_status' for c in COMPONENTS]].eq('ok').all(axis=1)
    delivered=~replay[[f'{c}_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
    trusted=independently_trusted(source,replay.target_timestamp)
    assert np.array_equal(replay.score_selected_only,selected & finite & baseline_finite & trusted)
    assert np.array_equal(replay.score_delivered,delivered & finite & baseline_finite & trusted)
    assert saved.score_saved_benchmark.all() and saved.trusted_target.all()
    masks={'saved_benchmark':(saved,np.ones(len(saved),dtype=bool)),
           'replay_selected_only':(replay,replay.score_selected_only), 'replay_delivered':(replay,replay.score_delivered)}
    metric_csv=pd.read_csv(directory/'metrics.csv').set_index('cohort')
    monthly=pd.read_csv(directory/'monthly_metrics.csv')
    for row in summary['cohorts']:
        frame,mask=masks[row['cohort']];scored=frame.loc[mask]
        assert len(scored)==row['rows']==metric_csv.loc[row['cohort'],'rows']
        measured,bootstrap=independent_metrics(scored)
        for key,value in measured.items():
            close(row[key],value);close(metric_csv.loc[row['cohort'],key],value)
        for key,value in bootstrap.items():
            close(row['bootstrap'][key],value);close(metric_csv.loc[row['cohort'],f'bootstrap_{key}'],value)
        close(row['coverage_nominal_pct'],100*len(scored)/len(replay))
        close(row['coverage_trusted_pct'],100*len(scored)/np.sum(trusted))
        for _,month in monthly.loc[monthly.cohort.eq(row['cohort'])].iterrows():
            in_month=frame.target_timestamp.dt.strftime('%Y-%m').eq(month['month'])
            monthly_scored=frame.loc[mask & in_month]
            measured,_=independent_metrics(monthly_scored)
            assert month['rows']==len(monthly_scored)
            for key,value in measured.items():close(month[key],value)
            n=int(replay.target_timestamp.dt.strftime('%Y-%m').eq(month['month']).sum())
            t=int((replay.target_timestamp.dt.strftime('%Y-%m').eq(month['month']) & trusted).sum())
            close(month.coverage_nominal_pct,100*len(monthly_scored)/n if n else None)
            close(month.coverage_trusted_pct,100*len(monthly_scored)/t if t else None)
    counts=summary['replay_counts']
    assert counts['trusted_targets']==int(np.sum(trusted))
    assert counts['target_excluded']==int((~trusted).sum())
    assert counts['delivered_scored']==int(replay.score_delivered.sum())
    assert counts['selected_only_scored']==int(replay.score_selected_only.sum())
    assert counts['blocked_any_component_all_origins']==int((~delivered).sum())
    assert counts['blocked_any_component_trusted_targets']==int((~delivered & trusted).sum())
    fallback=replay[[f'{c}_status' for c in COMPONENTS]].eq('fallback').any(axis=1)
    assert counts['fallback_rows_scored_delivered']==int((fallback & replay.score_delivered).sum())
    assert counts['any_fallback_all_origins']==int(fallback.sum())
    for component in COMPONENTS:
        assert counts['components'][component]['statuses']==replay[f'{component}_status'].value_counts().to_dict()
    assert summary['saved_alignment']['common_rows']==len(saved)
    assert summary['saved_alignment']['missing_patterns']==audit.saved_missing_pattern.value_counts().to_dict()
    return {'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'saved_rows':len(saved),'replayed_origins':len(replay),'trusted_targets':int(np.sum(trusted)),
            'runtime_versions':{'python':platform.python_version(),**{name:version(name) for name in ['numpy','pandas','xgboost','scikit-learn','scipy']}},
            'checks':['origin/target uniqueness and horizon','source actual/persistence alignment','saved forecast identity',
                      'signed energy balance and component error identity','independent retrospective target policy',
                      'identical model/baseline cohort masks','independent net/renewable/share metrics',
                      'independent day-bootstrap intervals','monthly metrics and coverage','fallback/blocked counts',
                      'source/model/code hashes'],
            'output_sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='verification.json'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    report=verify(args.root)
    output=args.root/'reports/experiments/grid_evaluation_v1/verification.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
