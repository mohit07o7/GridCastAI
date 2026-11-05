"""Reproduce causal 24-hour forecast scores and constrained battery simulations."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.curve_features import COMPONENTS,TEST_START,VALIDATION_START,CurveHistory
from common.curve_model import GridCurveForecaster,digest,json_write
from common.battery_dispatch import load_scenario,scenario_variants
from common.grid_operations import (PlanSink,attach_evaluation,forecast_scores,simulate,operations_scores)

ROOT=Path(__file__).resolve().parents[2]


def disabled(*args,**kwargs):raise AssertionError('Training is disabled during operational replay')


def evaluate_forecast_only(root):
    """Finish the forecast deliverable without invoking battery planning."""
    service=GridCurveForecaster(root,strict=True)
    meta=service.manifest
    source=root/meta['source']['path']
    if digest(source)!=meta['source']['sha256']:raise ValueError('Source hash mismatch')
    for path,sha in meta['code_sha256'].items():
        if digest(root/path)!=sha:raise ValueError(f'Changed curve training/feature code: {path}')
    raw=pd.read_csv(source,parse_dates=['timestamp'])
    context=CurveHistory(raw)
    origins=context.index[(context.index>=TEST_START) & (context.index<=context.index.max()-24*pd.Timedelta(hours=1))]
    selected=service.forecast_rows(context,origins)
    baseline=service.forecast_rows(context,origins,baseline_only=True)
    rows=attach_evaluation(selected,baseline,raw.set_index('timestamp'))
    metrics=forecast_scores(rows)
    directory=root/'reports/experiments/grid_curve_v1';directory.mkdir(exist_ok=True)
    rows.to_csv(directory/'forecast_rows.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    metrics.to_csv(directory/'forecast_metrics.csv',index=False,float_format='%.17g')
    status=selected[[c+'_status' for c in COMPONENTS]]
    complete=(~status.eq('blocked').any(axis=1)).groupby(selected.origin).all()
    fallbacks=status.eq('fallback').any(axis=1).groupby(selected.origin).any()
    summary={'schema_version':1,'run_context':'retrospective forecast evaluation; not an untouched holdout',
        'training_disabled':True,'battery_status':'deferred by user; no battery planning or verification in this run',
        'nominal_origins':len(origins),'forecast_exposures':len(rows),
        'origin_start':str(origins.min()),'origin_end':str(origins.max()),
        'target_start':str(rows.target_timestamp.min()),'target_end':str(rows.target_timestamp.max()),
        'complete_curves':int(complete.sum()),'curves_with_fallback':int(fallbacks.sum()),
        'component_status_exposures':{c:status[c+'_status'].value_counts().to_dict() for c in COMPONENTS},
        'inference_policy':meta['inference_policy'],'target_policy':'grid_evaluation_target_v1',
        'source':meta['source'],'model_manifest_sha256':digest(service.directory/'metadata.json'),
        'model_sha256':{k:e['model_sha256'] for k,e in meta['entries'].items()},
        'code_sha256':{str(p.relative_to(root)):digest(p) for p in [Path(__file__),
            root/'src/common/curve_features.py',root/'src/common/curve_model.py',
            root/'src/common/inference_history.py',root/'src/common/grid_evaluation.py',root/'src/common/grid_operations.py']},
        'versions':{'python':platform.python_version(),**{n:version(n) for n in ['numpy','pandas','xgboost','scipy']}}}
    json_write(directory/'summary.json',summary)
    headline=metrics[(metrics.month=='all') & (metrics.quantity=='net') & (metrics.cohort=='delivered')]
    lines=['# Verified 24-hour forecasting v1','',
        'Forecasting is complete. Battery implementation and verification are set aside at the user’s request. This report contains no battery result.','',
        'Results are retrospective: these historical periods have already informed development. The selected forecast and validation-selected baseline use identical trusted delivered hours within each comparison.','',
        '| Horizon | Paired hours | Net-demand forecast MAE | Baseline MAE | MAE reduction |',
        '|---|---:|---:|---:|---:|']
    for h in [1,6,12,24]:
        r=headline[headline.horizon.eq(h)].iloc[0]
        lines.append(f'| {h} | {int(r.rows)} | {r.forecast_mae_mw:,.2f} MW | {r.baseline_mae_mw:,.2f} MW | {r.mae_improvement_pct:.2f}% |')
    lines+=['',f'The delivered system lowers paired net-demand MAE at {int(headline.mae_improvement_mw.gt(0).sum())}/24 horizons. This does not establish future performance or any battery benefit.',
        f'All {len(origins):,} nominal issue times return a complete curve; {int(fallbacks.sum()):,} curves contain at least one labelled fallback.' if complete.all() else f'{int(complete.sum()):,}/{len(origins):,} issue times return complete curves; all missing components and reasons remain in the exported rows.',
        '', '## Monthly behavior','']
    for h in [1,24]:
        m=metrics[(metrics.month!='all') & metrics.quantity.eq('net') & metrics.cohort.eq('delivered') & metrics.horizon.eq(h)]
        losses=m.loc[m.mae_improvement_mw.lt(0),'month'].tolist()
        lines.append(f'- Horizon {h}: lower MAE in {int(m.mae_improvement_mw.gt(0).sum())}/{len(m)} months; worse months: {", ".join(losses) if losses else "none"}.')
    lines+=['','## How it works','',
        '- One issue returns 24 distinct hourly targets, origin + 1 through origin + 24. Source-clock timestamps remain timezone-naive, and the observation at the origin is assumed available.',
        '- All 72 candidates were trained on training data only. Fixed chronological boundaries, a 24-hour purge, validation early stopping and validation-only method selection precede all test scoring. Validation selected XGBoost for every component/horizon; test scores did not change those choices.',
        '- Lag/rolling features and target-calendar fields contain no future labels. Observed quality checks use evidence available by the origin; retrospective target exclusions attach only after generating predictions.',
        '- Baselines are last observation, target-aligned previous day and previous week. If a selected method is unavailable, eligible baselines are explicit fallbacks; unsafe components remain null.',
        '- The existing single-target command, older model bundles, source measurements and historical results remain unchanged. The new bundle stores native models, feature order/types, selected iterations, configuration, dates, row counts, versions and hashes.',
        '- Selected-only and delivered cohorts are exported separately. MAE, RMSE, WAPE and signed error (forecast minus actual) are reported by component, horizon and month. A zero WAPE denominator is undefined.',
        '', '## Reproduce','', '```bash',
        '.venv/bin/python src/forecasting/predict_grid_curve.py --origin "2025-06-24 23:00:00" --history data/processed/full_hourly_stacked.csv',
        '.venv/bin/python src/analysis/evaluate_grid_operations.py --forecast-only',
        '.venv/bin/python src/investigations/verify_grid_curve.py',
        '.venv/bin/python src/investigations/verify_grid_curve_evaluation.py',
        '```','',
        'forecast_rows.csv.gz contains every forecast, method, quality status and retrospective exclusion. forecast_metrics.csv contains every paired metric. verification.json verifies native model reproduction; evaluation_verification.json independently verifies exported scores. No training occurs in prediction/evaluation/verification.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    print(headline[headline.horizon.isin([1,6,12,24])].to_json(orient='records',indent=2))
    print(f'Forecast-only artifacts saved to {directory}')


def report_text(summary,forecast,operations):
    lines=['# Operational forecasting and battery simulation v1','',
        'Retrospective, deterministic simulation. These historical periods have already informed development. Prices and equipment are illustrative; this is not evidence of national operational dispatch.','',
        '## Forecasting result','',
        'Each issue produces horizons 1–24 from one observed history cutoff. Models are direct, training-only fits; validation chose the methods. Model and baseline scores use identical trusted delivered rows within each comparison.','',
        '| Horizon | Paired hours | Net-demand MAE | Baseline MAE | MAE improvement |',
        '|---|---:|---:|---:|---:|']
    selected=forecast[(forecast.month=='all') & (forecast.quantity=='net') & (forecast.cohort=='delivered')]
    for h in [1,6,12,24]:
        r=selected[selected.horizon.eq(h)].iloc[0]
        lines.append(f'| {h} | {int(r.rows)} | {r.forecast_mae_mw:,.2f} MW | {r.baseline_mae_mw:,.2f} MW | {r.mae_improvement_pct:.2f}% |')
    lines+=['','These are horizon-specific results, not the earlier single-hour experiment or a newly untouched holdout. Selected-only and delivered/fallback cohorts are exported separately.','',
        '## Realized battery result','',
        'The controller replans 24 hours but executes only hour one. Actuals reconcile that fixed action, and the resulting charge is carried forward. Reported realized costs differ from the optimizer’s predicted 24-hour costs.','',
        '| Policy | Electricity + wear cost | Unmet demand | Hours with unmet demand | Unavailable plans |',
        '|---|---:|---:|---:|---:|']
    main=operations[(operations.scenario=='operations_v1') & (operations.month=='all')].set_index('policy')
    for policy,r in main.iterrows():
        lines.append(f'| {policy} | ₹{r.realized_total_cost_inr:,.2f} | {r.unserved_mwh:,.3f} MWh | {int(r.unserved_hours)} | {int(r.unavailable_plan_hours)} |')
    a=main.loc['forecast_battery'];b=main.loc['baseline_battery']
    saving=b.realized_total_cost_inr-a.realized_total_cost_inr
    service=b.unserved_mwh-a.unserved_mwh
    lines+=['',f'Against baseline-driven battery scheduling, the selected forecasting controller changes total realized cost by ₹{-saving:,.2f} (negative means cheaper) and unmet demand by {-service:,.3f} MWh (negative means better service).',
        'Reliability has priority: a cheaper controller that leaves more demand unmet is not a reliability improvement. Costs are reported alongside service rather than treating withheld supply as a saving.']
    if service>1e-5:
        lines.append('The selected forecast controller serves more demand in this simulation; the cost comparison must account for that additional delivered energy.')
    elif service < -1e-5:
        lines.append('The selected forecast controller leaves more demand unmet. It does not meet the intended reliability improvement over the baseline controller.')
    else:lines.append('The controllers deliver essentially identical service in the default scenario.')
    months=operations[(operations.scenario=='operations_v1') & (operations.month!='all')]
    pivot=months.pivot(index='month',columns='policy',values='unserved_mwh')
    costpivot=months.pivot(index='month',columns='policy',values='realized_total_cost_inr')
    lines.append(f'Across {len(pivot)} scored months, forecasting reduces unmet demand in {int((pivot.baseline_battery-pivot.forecast_battery>1e-5).sum())}, worsens it in {int((pivot.baseline_battery-pivot.forecast_battery < -1e-5).sum())}; cost is lower in {int((costpivot.baseline_battery-costpivot.forecast_battery>1e-5).sum())}. Monthly results and sensitivity results remain separate comparisons.')
    lines+=['','## Assumptions, coverage and limits','',
        '- National demand, wind and solar use one common scale factor derived solely from the pre-validation training-period peak. Later values are not clipped to 100 MW.',
        '- Hourly power is held constant for one simulated hour. Synthetic prices follow the timezone-naive source clock; no timezone conversion or claim of real tariffs is made.',
        '- The 80 MWh battery uses 20 MW power, 90% round-trip efficiency, 10–90% operating charge, 50% initial charge and a 50% terminal reserve. Capacity sensitivities retain those fractions; power stays 20 MW.',
        '- Shortages remain visible. Export is disabled. Charging and discharging cannot occur together. All returned plans are independently checked for physical feasibility.',
        '- Target quality is unknown during forecasting. Full-series retrospective target flags are attached later for scoring and physical-episode auditing. Excluded actuals break simulation episodes; the next trusted episode restarts at 50% charge for every battery controller.',
        '- Plans are calculated before examining the current target’s actual/quality. Episode resets are a retrospective evaluation convention, not an operational state-estimation method.',
        '- Blocked forecasts or failed solves idle the battery. Their hours are retained; they are not removed to improve cost or reliability scores.',
        f"- Nominal curve origins: {summary['nominal_origins']}; complete selected curves: {summary['coverage']['complete_curves']}; curves containing a fallback: {summary['coverage']['curves_with_fallback']}.",
        f"- Default simulation trusted hours: {int(a.trusted_hours)}; excluded hours: {int(a.excluded_hours)}; trusted episodes: {int(a.episode_starts)}.",
        '- Aggregate recorded renewables are used as simulated availability, not independently measured unconstrained renewable potential. Network flows, generator commitment, real settlement rules and uncertainty scenarios are outside this version.','',
        '## Reproduce','',
        '```bash',
        '.venv/bin/python src/analysis/evaluate_grid_operations.py',
        '.venv/bin/python src/investigations/verify_grid_operations.py',
        '```','',
        'forecast_rows.csv.gz contains every origin/horizon, method, fallback and exclusion. dispatch_rows.csv.gz contains all scenario/controller hours, including excluded or unavailable hours. plans.csv.gz contains every successfully solved 24-hour plan. Metrics, monthly results, hashes and verification reports are saved alongside them. Existing measurements, model bundles and historical results are preserved.']
    return '\n'.join(lines)+'\n'


def evaluate(root):
    service=GridCurveForecaster(root,strict=True)
    meta=service.manifest
    source_path=root/meta['source']['path']
    if digest(source_path)!=meta['source']['sha256']:raise ValueError('Source hash differs from the trained curve bundle')
    for p,sha in meta['code_sha256'].items():
        if digest(root/p)!=sha:raise ValueError(f'Curve training/feature code changed: {p}')
    scenario=load_scenario('operations_v1',root)
    raw=pd.read_csv(source_path,parse_dates=['timestamp'])
    training_peak=float(raw.loc[raw.timestamp<VALIDATION_START,'demand_mw'].max())
    if abs(training_peak-scenario.training_peak_mw)>1e-8 or abs(scenario.scale_factor*training_peak-100)>1e-9:
        raise ValueError('Scenario scale is not the documented training-period 100 MW peak')
    context=CurveHistory(raw)
    origins=context.index[(context.index>=TEST_START) & (context.index<=context.index.max()-24*pd.Timedelta(hours=1))]
    print(f'Generating selected and baseline curves for {len(origins)} origins; training disabled',flush=True)
    selected=service.forecast_rows(context,origins)
    baseline=service.forecast_rows(context,origins,baseline_only=True)
    source=raw.set_index('timestamp')
    audited=attach_evaluation(selected,baseline,source)
    forecast=forecast_scores(audited)
    parent=root/'reports/experiments';parent.mkdir(exist_ok=True)
    destination=parent/'grid_operations_v1'
    curve_destination=parent/'grid_curve_v1';curve_destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='grid-operations-',dir=parent) as tmp:
        staging=Path(tmp)
        audited.to_csv(staging/'forecast_rows.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
        forecast.to_csv(staging/'forecast_metrics.csv',index=False,float_format='%.17g')
        dispatch=[]
        sink=PlanSink(staging/'plans.csv.gz')
        try:
            for variant in scenario_variants(scenario):
                print(f'Simulating {variant.scenario_id}',flush=True)
                frame=simulate(selected,baseline,source,variant,sink,
                    lambda i,n:print(f'{variant.scenario_id}: {i}/{n} hourly decisions',flush=True))
                dispatch.append(frame)
        finally:sink.close()
        delivered=pd.concat(dispatch,ignore_index=True)
        delivered.to_csv(staging/'dispatch_rows.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
        operations=operations_scores(delivered)
        operations.to_csv(staging/'operations_metrics.csv',index=False,float_format='%.17g')
        statuses=selected[[c+'_status' for c in COMPONENTS]]
        selected_delivery=~statuses.eq('blocked').any(axis=1)
        group=selected_delivery.groupby(selected.origin)
        fallback=statuses.eq('fallback').any(axis=1).groupby(selected.origin).any()
        summary={'schema_version':1,'run_context':'retrospective deterministic simulation; not operational dispatch',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'nominal_origins':len(origins),'forecast_exposures':len(audited),
            'origin_start':str(origins.min()),'origin_end':str(origins.max()),
            'executed_target_start':str(origins.min()+pd.Timedelta(hours=1)),
            'executed_target_end':str(origins.max()+pd.Timedelta(hours=1)),
            'coverage':{'complete_curves':int(group.all().sum()),'partial_curves':int((group.any() & ~group.all()).sum()),
                        'fully_blocked_curves':int((~group.any()).sum()),'curves_with_fallback':int(fallback.sum()),
                        'component_status_exposures':{c:statuses[c+'_status'].value_counts().to_dict() for c in COMPONENTS}},
            'policies':{'inference':meta['inference_policy'],'target':'grid_evaluation_target_v1',
                        'simulation':'receding_hourly_battery_v1','priority':'minimum unmet MWh, then electricity plus throughput wear cost'},
            'scenarios':[asdict(s) for s in scenario_variants(scenario)],
            'capacity_sensitivity_convention':'SOC fractions are fixed; initial/terminal/operating MWh scale with capacity; power remains 20 MW',
            'training_disabled':True,'models_changed':False,'plan_rows':sink.rows,
            'source':meta['source'],'model_manifest_sha256':digest(service.directory/'metadata.json'),
            'model_sha256':{k:e['model_sha256'] for k,e in meta['entries'].items()},
            'scenario_sha256':digest(root/'configs/operations_v1.json'),
            'code_sha256':{str(p.relative_to(root)):digest(p) for p in [
                root/'src/common/grid_operations.py',root/'src/common/battery_dispatch.py',
                root/'src/analysis/evaluate_grid_operations.py',root/'src/common/grid_evaluation.py',
                root/'src/common/curve_features.py',root/'src/common/curve_model.py']},
            'versions':{'python':platform.python_version(),**{n:version(n) for n in ['numpy','pandas','xgboost','scipy']}}}
        json_write(staging/'summary.json',summary)
        (staging/'report.md').write_text(report_text(summary,forecast,operations))
        destination.mkdir(exist_ok=True)
        for path in staging.iterdir():path.replace(destination/path.name)
    (curve_destination/'report.md').write_text('# Direct 24-hour curves v1\n\nTraining fitted 72 independent component/horizon XGBoost candidates on training data only. Validation selected XGBoost at all 72 component/horizon pairs; simple baselines remain available as explicit fallbacks. No validation/test refit occurred.\n\nThe adjacent verification report checks all 359,352 in-memory reference predictions in a fresh load-only process. Operational test metrics and complete forecast exclusions are in ../grid_operations_v1/. These are retrospective results.\n')
    print(operations[(operations.month=='all') & (operations.scenario=='operations_v1')].to_json(orient='records',indent=2),flush=True)
    print(f'Artifacts saved to {destination}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--forecast-only',action='store_true',help='Evaluate only the delivered forecast curves; do not plan batteries')
    args=parser.parse_args()
    with patch.object(xgb.XGBRegressor,'fit',disabled),patch.object(xgb,'train',disabled),patch.object(xgb.Booster,'update',disabled):
        if args.forecast_only:
            with patch('common.battery_dispatch.milp',side_effect=AssertionError('Battery work is deferred')):
                evaluate_forecast_only(args.root)
        else:evaluate(args.root)
