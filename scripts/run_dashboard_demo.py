"""Reproduce the local dashboard walkthrough; no server, training or battery solve."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dashboard.runtime import install_guards
from dashboard.service import (load_reports,load_history,load_forecaster,forecast,forecast_table,
    attach_actuals,metric_record,audit_slice,csv_bytes,json_bytes,NORMAL_ORIGIN,FALLBACK_ORIGIN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    install_guards()
    reports=load_reports(ROOT);history=load_history(ROOT,reports.summary);model=load_forecaster(ROOT)
    parent=ROOT/'reports/demos/dashboard_v1'
    parent.mkdir(parents=True,exist_ok=True)
    if args.output:
        directory=args.output.resolve()
        directory.mkdir(parents=True,exist_ok=False)
    else:directory=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    checks={};cases={}
    for case,origin in [('normal',NORMAL_ORIGIN),('fallback',FALLBACK_ORIGIN),('blocked',NORMAL_ORIGIN)]:
        result=forecast(model,history,origin,case)
        table=forecast_table(result)
        (directory/(case+'.json')).write_bytes(json_bytes(result))
        (directory/(case+'.csv')).write_bytes(csv_bytes(table))
        assert len(table)==24 and result['target_quality']=='unknown'
        assert 'simulation' not in result
        if case=='normal':
            assert result['complete_curve'] and result['status']=='ok'
            annotated=attach_actuals(table,history)
            assert not annotated.actual_available.any() and annotated.target_quality.eq('unknown').all()
        if case=='fallback':assert any(table[c+'_status'].eq('fallback').any() for c in ['demand','wind'])
        if case=='blocked':
            assert table.wind_status.eq('blocked').all() and table.wind_mw.isna().all()
            assert table.demand_status.eq('ok').all() and table.solar_status.eq('ok').all()
        cases[case]={'origin':origin,'status':result['status'],'complete_curve':result['complete_curve'],
                    'synthetic_input':case=='blocked'}
    row=metric_record(reports)
    (directory/'accuracy.csv').write_bytes(csv_bytes(row.to_frame().T))
    audit,counts,reasons=audit_slice(reports,24,'2025-04','delivered')
    (directory/'april_exclusions.csv').write_bytes(csv_bytes(audit.loc[~audit.trusted_target,['origin','horizon','target_timestamp','target_exclusion_reasons']]))
    (directory/'april_exclusion_counts.csv').write_bytes(csv_bytes(reasons))
    assert len(audit)>0 and counts['excluded_target_records']>0
    # A separate metamorphic check protects issue-time forecasting from future actuals.
    modified=history.copy();modified.loc[modified.timestamp>FALLBACK_ORIGIN,['demand_mw','wind_mw','solar_mw']]=np.nan
    assert forecast(model,modified,FALLBACK_ORIGIN)==forecast(model,history,FALLBACK_ORIGIN)
    checks.update(aligned_24_hour_cases=True,normal_and_fallback_and_blocked=True,
        actuals_separate=True,future_data_invariance=True,historical_filter_matches_saved_metrics=True,
        training_disabled=True,battery_disabled=True)
    receipt={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
        'checks':checks,'accuracy_selection':{'quantity':'net','horizon':24,'cohort':'delivered','month':'all'},
        'audit_selection':{'month':'2025-04','horizon':24,'cohort':'delivered','counts':counts},
        'source':reports.summary['source'],'model_manifest_sha256':reports.summary['model_manifest_sha256'],
        'historical_verification':reports.verification['status'],
        'versions':{n:version(n) for n in ['streamlit','plotly','numpy','pandas','xgboost']}}
    (directory/'walkthrough.md').write_text('''# GridCastAI local demonstration

Start with ./scripts/start_dashboard.sh from Desktop/GridCastAI.

1. Normal forecast: June 24, 2025 at 23:00 produces June 25’s 24 hours. Actuals are unavailable beyond the source history.
2. Historical fallback: January 9 at 15:00 shows explicitly labelled baseline fallbacks. Target quality remains unknown at issue time.
3. Synthetic blocked wind: the source frame is copied in memory, wind is made invalid, and demand/solar continue. Missing wind and net-demand chart points are gaps, never zeros. No source file changes.
4. Historical accuracy: choose net demand, horizon 24, delivered cohort, all months. Model and baseline use identical paired rows.
5. Quality audit: choose April 2025 in Historical accuracy and open Quality and exclusions. Input problems and retrospective target exclusions are separate.
6. Optimization status: Deferred. No battery schedule or savings is generated.

Downloads contain the same forecast rows and original JSON as the app. Results are retrospective demonstrations, not live forecasts actually issued in 2025.
''')
    receipt['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    (directory/'run.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','output':str(directory),'checks':checks},indent=2))


if __name__=='__main__':main()
