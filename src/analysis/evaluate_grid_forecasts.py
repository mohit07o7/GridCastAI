"""Rebuild saved and command-replay net-demand evaluations without training."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from collections import Counter

import numpy as np
import pandas as pd
import xgboost
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid_prediction import GridForecaster, ROOT
from common.demand_model import sha256
from common.features import build_features
from common.splits import chronological_split
from common.grid_evaluation import (
    COMPONENTS, KEYS, EVALUATION_POLICY, validate_component, validate_keys,
    attach_observations, combined_quantities, retrospective_target_flags,
    attach_target_flags, assign_replay_cohorts, paired_summary,
)
from forecasting.renewable_v3_full_history import build_frame, frozen_run_mask

FILES = {'demand':'xgboost_demand_predictions.csv', 'wind':'renewable_v3_wind_predictions.csv',
         'solar':'renewable_v3_solar_predictions.csv'}


def load_saved(root, source, engine):
    tables = {}
    paths = []
    for component, filename in FILES.items():
        path = root/'reports/experiments'/filename
        paths.append(path)
        frame = pd.read_csv(path)
        if component == 'demand':
            frame = frame.rename(columns={'Forecast Origin':'origin','Target Timestamp':'target_timestamp',
                'Actual Demand (MW)':'actual_demand','XGBoost Forecast (MW)':'predicted_demand',
                'Seasonal Naive Forecast (MW)':'baseline_demand'})
        else:
            frame = frame.rename(columns={'actual':f'actual_{component}', 'selected':f'predicted_{component}',
                                          'prior_day':f'baseline_{component}'})
            selected = frame[f'predicted_{component}']
            expected = frame.residual_xgb if component == 'wind' else frame[f'baseline_{component}']
            if not np.allclose(selected,expected,rtol=0,atol=1e-9):
                raise ValueError(f'{component}: CSV does not use the selected method')
        keep = KEYS + [f'{p}_{component}' for p in ['actual','predicted','baseline']]
        frame = validate_component(frame[keep], source, component)
        cutoff = pd.Timestamp(engine.entries[component]['latest_training_or_tuning_target'])
        if frame.origin.lt(cutoff).any():
            raise ValueError(f'{component}: saved prediction precedes latest training/tuning target')
        tables[component] = frame
    # Verify demand belongs to the current saved model bundle, not a historical run.
    reference = pd.read_csv(root/'models/demand_v1/test_reference.csv').rename(columns={
        'Forecast Origin':'origin','Target Timestamp':'target_timestamp','XGBoost Forecast (MW)':'reference_prediction'})
    reference = validate_keys(reference, 'model reference')
    comparison = tables['demand'].merge(reference[KEYS+['reference_prediction']],on=KEYS,how='outer',validate='one_to_one',indicator=True)
    if not comparison['_merge'].eq('both').all() or not np.allclose(comparison.predicted_demand,comparison.reference_prediction,rtol=0,atol=.02):
        raise ValueError('Demand predictions do not match the current model reference')
    return tables, paths


def _keyset(frame):
    return set(pd.MultiIndex.from_frame(frame[KEYS]).tolist())


def reconstruct_eligibility(raw, tables, engine, nominal):
    """Reproduce existing historical eligibility; do not change training rules."""
    demand = build_features(raw.rename(columns={'timestamp':'Timestamp'}))
    names = engine.entries['demand']['features']
    _, _, test = chronological_split(demand.dropna(subset=names+['Target']).reset_index(drop=True))
    expected = test[['Timestamp','Target Timestamp']].rename(columns={'Timestamp':'origin','Target Timestamp':'target_timestamp'})
    if _keyset(expected) != _keyset(tables['demand']):
        raise ValueError('Saved demand rows do not match current split/quality policy')
    audit = nominal.copy()
    demand.index = pd.DatetimeIndex(demand.Timestamp)
    reasons = {'demand':pd.DataFrame({'target_quality':demand.quality_target_bad,'input_quality':demand.quality_input_bad})}
    bad = raw[['demand_mw','wind_mw','solar_mw','total_gen_mw']].eq(0).all(axis=1)
    bad |= (raw.solar_wind_mw-raw.wind_mw-raw.solar_mw).abs().gt(1)
    solar_flatline = frozen_run_mask(raw.solar_mw) & raw.solar_mw.gt(500)
    for component in ['wind','solar']:
        invalid = bad | solar_flatline if component=='solar' else bad
        frame, _ = build_frame(raw,f'{component}_mw',invalid)
        expected = frame.loc[frame.origin_position.ge(int(len(raw)*.85)),KEYS]
        if _keyset(expected) != _keyset(tables[component]):
            raise ValueError(f'Saved {component} rows do not match v3 eligibility')
        flags = pd.DataFrame({'target_quality':invalid.shift(-24,fill_value=False),
                              'input_quality':invalid.rolling(168,min_periods=1).max().astype(bool)
                                  | invalid.shift(168,fill_value=False) | invalid.shift(8736,fill_value=False)})
        flags.index = pd.DatetimeIndex(raw.timestamp)
        reasons[component]=flags
    for component in COMPONENTS:
        presence = tables[component][KEYS].assign(present=True)
        audit = audit.merge(presence,on=KEYS,how='left',validate='one_to_one')
        audit[f'{component}_saved_present'] = audit.pop('present').fillna(False).astype(bool)
        flag = reasons[component].reindex(pd.DatetimeIndex(audit.origin)).fillna(False)
        per_row=[]
        for present, (_,row) in zip(audit[f'{component}_saved_present'],flag.iterrows()):
            active=[] if present else [f'{component}_{name}' for name,value in row.items() if value]
            if not present and not active:
                active=[f'{component}_outside_test_or_incomplete_features']
            per_row.append(json.dumps(active))
        audit[f'{component}_saved_exclusion_reasons']=per_row
    audit['saved_common'] = audit[[f'{c}_saved_present' for c in COMPONENTS]].all(axis=1)
    audit['saved_missing_pattern'] = audit.apply(lambda row: json.dumps([c for c in COMPONENTS if not row[f'{c}_saved_present']]),axis=1)
    return audit


def replay_predictions(raw, nominal, engine):
    """No actual target readings or evaluation flags are attached in this phase."""
    rows=[]
    for count, origin in enumerate(nominal.origin,start=1):
        report=engine.forecast(raw,origin)
        if report['target_quality']!='unknown' or pd.Timestamp(report['target_timestamp'])!=origin+pd.Timedelta(hours=24):
            raise ValueError('Replay target or inference target-quality contract changed')
        row={'origin':origin,'target_timestamp':pd.Timestamp(report['target_timestamp']),
             'inference_status':report['status'],'inference_target_quality':report['target_quality'],
             'inference_run_context':report['run_context']}
        for component,value in report['components'].items():
            if value['target_timestamp']!=report['target_timestamp']:
                raise ValueError('Component target timestamp mismatch')
            if any('artifact_unavailable' in reason for reason in value['reasons']):
                raise ValueError('Artifact failed during replay; do not score a broken deployment')
            row[f'predicted_{component}']=value['forecast_mw']
            for field in ['status','method','selected_model_id','selected_artifact_sha256']:
                row[f'{component}_{field}']=value[field]
            for field in ['reasons','unavailable_quality_checks']:
                row[f'{component}_{field}']=json.dumps(value[field])
        rows.append(row)
        if count%250==0 or count==len(nominal):
            print(f'Replayed {count}/{len(nominal)} origins',flush=True)
    return validate_keys(pd.DataFrame(rows),'replay')


def compare_saved_replay(saved, replay):
    merged=saved.merge(replay,on=KEYS,suffixes=('_saved','_replay'),validate='one_to_one')
    result={}
    all_selected=merged[[f'{c}_status' for c in COMPONENTS]].eq('ok').all(axis=1)
    for component in COMPONENTS:
        selected=merged[f'{component}_status'].eq('ok')
        difference=(merged.loc[selected,f'predicted_{component}_saved']-merged.loc[selected,f'predicted_{component}_replay']).abs()
        if not difference.le(.02).all():
            raise ValueError(f'{component}: replay differs from saved selected predictions')
        result[component]={'selected_rows_compared':int(selected.sum()),
                           'max_difference_mw':float(difference.max()) if len(difference) else None}
    result['all_components_selected_common_rows']=int(all_selected.sum())
    return result


def _flatten(summary):
    return {**{k:v for k,v in summary.items() if k!='bootstrap'},
            **{f'bootstrap_{k}':v for k,v in summary.get('bootstrap',{}).items()}}


def _counts(replay):
    result={'nominal_origins':len(replay),'trusted_targets':int(replay.trusted_target.sum()),
            'target_excluded':int((~replay.trusted_target).sum()),
            'selected_only_scored':int(replay.score_selected_only.sum()),'delivered_scored':int(replay.score_delivered.sum()),
            'any_fallback_all_origins':int(replay.any_fallback.sum()),
            'fallback_rows_scored_delivered':int((replay.any_fallback & replay.score_delivered).sum()),
            'fallback_frequency_delivered_pct':float(100*(replay.any_fallback & replay.score_delivered).sum()/replay.score_delivered.sum()) if replay.score_delivered.any() else None,
            'blocked_any_component_all_origins':int((~replay.delivered_forecast_available).sum()),
            'blocked_any_component_trusted_targets':int((~replay.delivered_forecast_available & replay.trusted_target).sum()),
            'target_exclusion_flags':{},'components':{}}
    for column in replay:
        if column.startswith('target_') and column not in ['target_timestamp','target_exclusion_reasons']:
            result['target_exclusion_flags'][column]=int(replay[column].sum())
    for component in COMPONENTS:
        reason_counts=Counter(reason for encoded in replay[f'{component}_reasons'] for reason in json.loads(encoded))
        unavailable=Counter(reason for encoded in replay[f'{component}_unavailable_quality_checks'] for reason in json.loads(encoded))
        result['components'][component]={'statuses':replay[f'{component}_status'].value_counts().to_dict(),
                                         'reason_counts':dict(reason_counts),'unavailable_check_counts':dict(unavailable)}
    return result


def write_report(summary, monthly, path):
    lines=['# Combined grid evaluation v1', '',
           'This is a retrospective evaluation. These historical periods have already informed model and quality-policy development.', '',
           '## What was compared', '',
           'Saved forecasts were joined by both origin and target timestamp. Command replay used the same saved models, history-only quality checks, and labelled fallbacks. Both methods and persistence always share the same scoring rows within each cohort.', '',
           'Net demand = demand − wind − solar. Renewable contribution is wind + solar, with its share expressed relative to demand. Net demand remains signed; this is a residual supply requirement, not a dispatch or battery optimization.', '',
           '| Cohort | Scored hours | Forecast MAE (MW) | Persistence MAE (MW) | MAE reduction |',
           '|---|---:|---:|---:|---:|']
    for row in summary['cohorts']:
        def fmt(value): return f'{value:,.2f}' if value is not None else 'undefined'
        lines.append(f"| {row['cohort']} | {row['rows']} | {fmt(row['predicted_net_mae_mw'])} | {fmt(row['baseline_net_mae_mw'])} | {fmt(row['mae_improvement_pct'])}% |")
    lines += ['', '## What the result means', '']
    for row in summary['cohorts']:
        if row['rows']==0:
            lines.append(f"- {row['cohort']}: no eligible paired rows; no accuracy conclusion is available.")
            continue
        change=row['mae_improvement_mw']; interval=row['bootstrap']
        direction='lower' if change>0 else ('higher' if change<0 else 'equal')
        months=monthly.loc[(monthly.cohort==row['cohort']) & monthly.rows.gt(0)]
        wins=int(months.mae_improvement_mw.gt(0).sum()); losses=int(months.mae_improvement_mw.lt(0).sum())
        confidence='favors the combined forecast' if interval['ci95_low_mw']>0 else ('favors persistence' if interval['ci95_high_mw']<0 else 'includes zero; the improvement is inconclusive under this resampling check')
        lines.append(f"- {row['cohort']}: forecast MAE is {abs(change):,.2f} MW {direction} than persistence. The approximate paired day-bootstrap 95% interval for MAE improvement is [{interval['ci95_low_mw']:,.2f}, {interval['ci95_high_mw']:,.2f}] MW and {confidence}. It improves in {wins}/{len(months)} scored months and worsens in {losses}; {'the direction is consistent across scored months' if wins==len(months) or losses==len(months) else 'the benefit is not consistent across months'}.")
    counts=summary['replay_counts']
    lines += ['', '## Coverage and exclusions', '',
              f"- Nominal test window: {summary['nominal']['rows']} origins, {summary['nominal']['first_origin']} through {summary['nominal']['last_origin']}.",
              f"- Saved common rows: {summary['saved_alignment']['common_rows']}; missing-pattern counts are {summary['saved_alignment']['missing_patterns']}.",
              f"- Replay trusted targets: {counts['trusted_targets']}; target exclusions: {counts['target_excluded']}.",
              f"- Delivered forecasts scored: {counts['delivered_scored']}; scored rows using a fallback: {counts['fallback_rows_scored_delivered']}.",
              f"- Any blocked component: {counts['blocked_any_component_all_origins']} origins, including {counts['blocked_any_component_trusted_targets']} trusted-target origins.", '',
              'The target policy excludes invalid/nonpositive-demand actuals, all-zero records, the April 4 demand anomaly, frozen demand/wind targets, positive solar flatlines and recorded renewable inconsistencies. Full-series retrospective flags are attached only after prediction generation. Other unclassified sharp changes are not automatically removed from target scoring.', '',
              'Selected-only and delivered replay scores use different row sets; do not rank them by their headline MAE alone. Coverage and monthly results show which hours each serves. The saved benchmark retains its inherited experiment exclusions. The demand prediction CSV has up to 0.00625 MW formatting differences from full-precision model outputs; the replay comparison allows 0.02 MW.', '',
              'All excluded origins and reasons are in alignment_audit.csv and replay_rows.csv. Missing component actuals are excluded from target scoring; a blocked forecast remains missing, never silently zero. Undefined renewable-share percentages are recorded and paired share metrics use a common valid-share mask.', '',
              'Metrics use signed error = forecast − actual. Positive MAE improvement means the combined forecast is better. WAPE uses the sum of absolute actual net demand, avoiding per-hour division near zero. Bootstrap resamples target calendar-day blocks (1,000 draws, seed 42); it is an exploratory interval, not proof of future or cross-region generalization.', '',
              '## Reproduce', '', '```bash', '.venv/bin/python src/analysis/evaluate_grid_forecasts.py',
              '.venv/bin/python src/investigations/verify_grid_evaluation.py', '```', '',
              'No training, source measurement changes or model replacement occurs. summary.json records artifact/data/code hashes, policies, exclusion counts and paired metrics. monthly_metrics.csv records month-by-month scores and coverage. verification.json contains independent exported-row checks.', '']
    path.write_text('\n'.join(lines))


def evaluate(root, destination):
    engine=GridForecaster(root)
    if engine.artifact_errors or set(engine.models)!= {'demand','wind'}:
        raise ValueError(f'Artifact preflight failed: {engine.artifact_errors}')
    solar_source=engine.entries['solar']['selection_source']
    if sha256(root/solar_source['path']) != solar_source['sha256']:
        raise ValueError('Solar selection provenance hash mismatch')
    renewable_metadata=json.loads((root/solar_source['path']).read_text())
    if any(renewable_metadata['targets']['solar']['selection'][period]!='prior_day' for period in ['06-20','21-05']):
        raise ValueError('Solar selected method changed')
    source_path=root/'data/processed/full_hourly_stacked.csv'
    raw=pd.read_csv(source_path,parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if raw.timestamp.isna().any() or raw.timestamp.duplicated().any() or raw.timestamp.diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError('Source must contain unique continuous hourly timestamps')
    source=raw.set_index('timestamp')
    tables,paths=load_saved(root,source,engine)
    first=max(f.origin.min() for f in tables.values()); last=min(f.origin.max() for f in tables.values())
    nominal=pd.DataFrame({'origin':pd.date_range(first,last,freq='h')})
    nominal['target_timestamp']=nominal.origin+pd.Timedelta(hours=24)
    audit=reconstruct_eligibility(raw,tables,engine,nominal)
    saved=tables['demand'].merge(tables['wind'],on=KEYS,validate='one_to_one').merge(tables['solar'],on=KEYS,validate='one_to_one')
    if _keyset(saved) != _keyset(audit.loc[audit.saved_common]):
        raise ValueError('Saved join and exclusion audit disagree')
    # Generate all predictions before computing/attaching retrospective scoring flags.
    replay=replay_predictions(raw,nominal,engine)
    parity=compare_saved_replay(saved,replay)
    target_flags=retrospective_target_flags(source)
    replay=assign_replay_cohorts(attach_target_flags(attach_observations(replay,source),target_flags))
    replay=combined_quantities(replay)
    audit=attach_target_flags(audit,target_flags)
    saved=combined_quantities(attach_target_flags(saved,target_flags))
    if not saved.trusted_target.all():
        raise ValueError('Saved common targets unexpectedly violate evaluation policy; investigate without silently filtering')
    saved['score_saved_benchmark']=True
    audit=audit.merge(replay[KEYS+['inference_status','selected_methods_available','delivered_forecast_available','any_fallback',
                                     'score_selected_only','score_delivered']],on=KEYS,validate='one_to_one')
    trusted=int(replay.trusted_target.sum())
    definitions=[('saved_benchmark',saved,saved.score_saved_benchmark),
                 ('replay_selected_only',replay,replay.score_selected_only),('replay_delivered',replay,replay.score_delivered)]
    cohorts=[paired_summary(f,mask,label,len(nominal),trusted) for label,f,mask in definitions]
    monthly=[]
    for month in nominal.target_timestamp.dt.to_period('M').unique():
        total=nominal.target_timestamp.dt.to_period('M').eq(month).sum()
        trust=(replay.target_timestamp.dt.to_period('M').eq(month)&replay.trusted_target).sum()
        for label,f,mask in definitions:
            m=paired_summary(f,mask & f.target_timestamp.dt.to_period('M').eq(month),label,int(total),int(trust),bootstrap=False)
            m['month']=str(month)
            if label.startswith('replay'):
                in_month=replay.target_timestamp.dt.to_period('M').eq(month)
                m['fallback_rows_scored']=int((mask & in_month & replay.any_fallback).sum())
                m['blocked_trusted_target_rows']=int((in_month & replay.trusted_target & ~replay.delivered_forecast_available).sum())
            monthly.append(m)
    monthly=pd.DataFrame(monthly)
    counts=_counts(replay)
    reasons={c:dict(Counter(reason for encoded in audit[f'{c}_saved_exclusion_reasons'] for reason in json.loads(encoded))) for c in COMPONENTS}
    artifacts=[source_path,*paths,root/'models/forecast_manifest.json',root/'models/demand_v1/model.json',
               root/'models/demand_v1/metadata.json',root/'models/demand_v1/test_reference.csv',root/'models/renewable_v3_wind.json',root/solar_source['path']]
    code=[Path(__file__),root/'src/common/grid_evaluation.py',root/'src/common/grid_prediction.py',root/'src/common/inference_history.py',
          root/'src/common/features.py',root/'src/common/splits.py',root/'src/common/demand_model.py',root/'src/forecasting/renewable_v3_full_history.py']
    summary={'schema_version':1,'evaluation_policy':EVALUATION_POLICY,'inference_policy':engine.manifest['inference_quality_policy'],
             'run_context':'retrospective; not an untouched holdout or operational dispatch validation',
             'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_disabled':True,
             'nominal':{'rows':len(nominal),'first_origin':str(first),'last_origin':str(last),
                        'first_target':str(first+pd.Timedelta(hours=24)),'last_target':str(last+pd.Timedelta(hours=24))},
             'saved_alignment':{'component_total_rows':{c:len(f) for c,f in tables.items()},'common_rows':len(saved),
                                'component_present_in_nominal':{c:int(audit[f'{c}_saved_present'].sum()) for c in COMPONENTS},
                                'missing_patterns':audit.saved_missing_pattern.value_counts().to_dict(),'overlapping_exclusion_reason_counts':reasons},
             'replay_counts':counts,'saved_replay_parity':parity,'cohorts':cohorts,
             'source_artifact_sha256':{str(p.relative_to(root)):sha256(p) for p in artifacts},
             'code_sha256':{str(p.relative_to(root)):sha256(p) for p in code},
             'versions':engine.models['demand'].metadata['versions']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staging=Path(directory)
        for name,frame in [('saved_rows',saved),('replay_rows',replay),('alignment_audit',audit),
                           ('metrics',pd.DataFrame([_flatten(c) for c in cohorts])),('monthly_metrics',monthly)]:
            frame.to_csv(staging/f'{name}.csv',index=False)
        (staging/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        write_report(summary,monthly,staging/'report.md')
        destination.mkdir(parents=True,exist_ok=True)
        for path in staging.iterdir():
            path.replace(destination/path.name)
    print('Saved evaluation:',destination,flush=True)
    for cohort in cohorts:
        print(cohort['cohort'],'rows=',cohort['rows'],'forecast MAE=',cohort['predicted_net_mae_mw'],
              'persistence MAE=',cohort['baseline_net_mae_mw'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    root=args.root.resolve()
    with ExitStack() as stack:
        for obj,name in [(xgboost.XGBRegressor,'fit'),(xgboost,'train'),(xgboost.Booster,'update')]:
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden during evaluation')))
        evaluate(root,root/'reports/experiments/grid_evaluation_v1')


if __name__=='__main__': main()
