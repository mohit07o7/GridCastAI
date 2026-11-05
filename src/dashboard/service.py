"""Read-only presentation service for verified forecasts and historical reports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.curve_model import GridCurveForecaster, digest

ROOT=Path(__file__).resolve().parents[2]
COMPONENTS=('demand','wind','solar')
NORMAL_ORIGIN='2025-06-24 23:00:00'
FALLBACK_ORIGIN='2025-01-09 15:00:00'
REPORT_DIR='reports/experiments/grid_curve_v1'


def signatures(paths):
    """Cheap cache keys; contents are hashed again on each cache miss."""
    result=[]
    for path in paths:
        p=Path(path)
        try:s=p.stat();result.append((str(p),s.st_size,s.st_mtime_ns))
        except FileNotFoundError:result.append((str(p),None,None))
    return tuple(result)


def model_paths(root):
    folder=Path(root)/'models/grid_curve_v1'
    return [folder/'metadata.json',*[folder/f'{c}_h{h:02d}.json' for c in COMPONENTS for h in range(1,25)]]


def report_paths(root):
    folder=Path(root)/REPORT_DIR
    return [folder/n for n in ['summary.json','forecast_metrics.csv','forecast_rows.csv.gz','evaluation_verification.json']]


@dataclass
class Reports:
    summary: dict
    verification: dict
    metrics: pd.DataFrame
    rows: pd.DataFrame


def load_reports(root=ROOT):
    directory=Path(root)/REPORT_DIR
    verification=json.loads((directory/'evaluation_verification.json').read_text())
    if verification.get('status')!='passed':raise ValueError('Historical evaluation is not verified')
    for name in ['summary.json','forecast_metrics.csv','forecast_rows.csv.gz']:
        if digest(directory/name)!=verification['output_sha256'][name]:
            raise ValueError(f'Historical artifact hash mismatch: {name}')
    summary=json.loads((directory/'summary.json').read_text())
    metrics=pd.read_csv(directory/'forecast_metrics.csv')
    rows=pd.read_csv(directory/'forecast_rows.csv.gz',parse_dates=['origin','target_timestamp'],float_precision='round_trip')
    needed={'origin','target_timestamp','horizon','trusted_target','target_exclusion_reasons',
            'score_selected_only','score_delivered','any_fallback'}
    needed|={f'{c}_{field}' for c in COMPONENTS for field in ['mw','method','status','reasons','unavailable_checks']}
    if not needed.issubset(rows):raise ValueError('Historical rows have missing required columns')
    required={'quantity','horizon','cohort','month','rows','nominal_rows','trusted_targets','coverage_trusted_pct',
              'forecast_mae_mw','baseline_mae_mw','forecast_rmse_mw','forecast_wape_pct','forecast_mean_signed_error_mw'}
    if not required.issubset(metrics):raise ValueError('Historical metrics have missing required columns')
    if rows.duplicated(['origin','horizon']).any() or metrics.duplicated(['quantity','horizon','cohort','month']).any():
        raise ValueError('Historical artifact contains duplicate keys')
    if not (rows.target_timestamp-rows.origin).eq(pd.to_timedelta(rows.horizon,unit='h')).all():
        raise ValueError('Historical targets are misaligned')
    return Reports(summary,verification,metrics,rows)


def load_history(root=ROOT,summary=None):
    root=Path(root)
    if summary is None:summary=json.loads((root/REPORT_DIR/'summary.json').read_text())
    path=root/summary['source']['path']
    if digest(path)!=summary['source']['sha256']:raise ValueError('Source history hash mismatch')
    data=pd.read_csv(path,parse_dates=['timestamp'])
    if not {'timestamp','demand_mw','wind_mw','solar_mw'}.issubset(data):raise ValueError('Missing source-history columns')
    return data


def load_forecaster(root=ROOT,expected_manifest=None):
    root=Path(root)
    if expected_manifest is None:
        expected_manifest=json.loads((root/REPORT_DIR/'summary.json').read_text())['model_manifest_sha256']
    if digest(root/'models/grid_curve_v1/metadata.json')!=expected_manifest:
        raise ValueError('Current forecast bundle differs from the verified experiment')
    return GridCurveForecaster(root,strict=True)


def forecast(forecaster,history,origin,case='normal'):
    if case not in ['normal','fallback','blocked']:raise ValueError('Unknown demonstration case')
    inputs=history
    if case=='blocked':
        # Explicit synthetic fixture; do not mutate the source frame or save it.
        inputs=history.loc[history.timestamp<=pd.Timestamp(origin)].tail(169).drop(columns=['solar_wind_mw'],errors='ignore').copy()
        inputs['wind_mw']=-1.
    return forecaster.forecast(inputs,origin)


def forecast_table(result):
    points=result['points']
    origin=pd.Timestamp(result['origin'])
    if len(points)!=24 or [p['horizon'] for p in points]!=list(range(1,25)):
        raise ValueError('Expected exactly 24 ordered horizons')
    rows=[]
    for p in points:
        if pd.Timestamp(p['target_timestamp'])!=origin+pd.Timedelta(hours=p['horizon']):
            raise ValueError('Forecast target misalignment')
        row={'origin':str(origin),'horizon':p['horizon'],'target_timestamp':pd.Timestamp(p['target_timestamp']),
             'target_quality':'unknown'}
        for c in COMPONENTS:
            entry=p['components'][c]
            v=entry['forecast_mw'];status=entry['status']
            if status not in ['ok','fallback','blocked']:raise ValueError('Unknown forecast status')
            if status=='blocked' and v is not None:raise ValueError('Blocked forecast must be null')
            if status!='blocked' and (v is None or not np.isfinite(v) or v<0):raise ValueError('Invalid delivered forecast')
            row[c+'_mw']=np.nan if v is None else float(v)
            row[c+'_method']=entry['method'];row[c+'_status']=status
            row[c+'_reasons']=json.dumps(entry['reasons'])
            row[c+'_unavailable_checks']=json.dumps(entry['unavailable_quality_checks'])
        row['renewable_mw']=row['wind_mw']+row['solar_mw']
        row['net_demand_mw']=row['demand_mw']-row['renewable_mw']
        row['renewable_share_pct']=100*row['renewable_mw']/row['demand_mw'] if row['demand_mw']>0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def attach_actuals(table,history):
    """Presentation only: run after forecasting; never modify target_quality."""
    result=table.copy()
    source=history.set_index('timestamp')
    for c in COMPONENTS:
        result['actual_'+c+'_mw']=source[c+'_mw'].reindex(result.target_timestamp).to_numpy()
    result['actual_available']=np.isfinite(result[['actual_'+c+'_mw' for c in COMPONENTS]]).all(axis=1)
    result['actual_renewable_mw']=result.actual_wind_mw+result.actual_solar_mw
    result['actual_net_demand_mw']=result.actual_demand_mw-result.actual_renewable_mw
    result['actual_renewable_share_pct']=100*result.actual_renewable_mw/result.actual_demand_mw.where(result.actual_demand_mw.gt(0))
    result['renewable_share_error_pp']=result.renewable_share_pct-result.actual_renewable_share_pct
    return result


def metric_record(reports,quantity='net',horizon=24,cohort='delivered',month='all'):
    m=reports.metrics
    found=m[m.quantity.eq(quantity)&m.horizon.eq(horizon)&m.cohort.eq(cohort)&m.month.eq(month)]
    if len(found)!=1:raise ValueError('No unique saved metric matches these filters')
    return found.iloc[0]


def audit_slice(reports,horizon=24,month='all',cohort='delivered'):
    if cohort not in ['delivered','selected_only']:raise ValueError('Unknown historical cohort')
    r=reports.rows
    selected=r[r.horizon.eq(horizon)].copy()
    if month!='all':selected=selected[selected.target_timestamp.dt.to_period('M').astype(str).eq(month)]
    scored=selected['score_'+cohort]
    counts={'forecast_records':len(selected),'unique_target_hours':selected.target_timestamp.nunique(),
            'trusted_records':int(selected.trusted_target.sum()),'scored_records':int(scored.sum()),
            'scored_unique_target_hours':selected.loc[scored,'target_timestamp'].nunique(),
            'excluded_target_records':int((~selected.trusted_target).sum()),
            'trusted_unscored_records':int((selected.trusted_target & ~scored).sum()),
            'fallback_records':int(selected.any_fallback.sum())}
    reasons=selected.loc[~selected.trusted_target,'target_exclusion_reasons'].map(json.loads).explode().dropna().value_counts()
    return selected,counts,reasons.rename_axis('reason').reset_index(name='forecast_records')


def csv_bytes(frame):
    return frame.to_csv(index=False,float_format='%.17g').encode()


def json_bytes(result):
    return (json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
