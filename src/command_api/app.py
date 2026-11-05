"""Local ASGI API and production React static server; never trains or dispatches."""
from collections import deque
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from services.grid_service import (ROOT, COMPONENTS, REPORT_DIR, signatures, model_paths, report_paths,
    load_reports, load_history, load_forecaster, forecast, forecast_table, metric_record, audit_slice, csv_bytes)
from common.inference_history import parse_origin
from common.grid_evaluation import retrospective_target_flags


class ResourceUnavailable(RuntimeError):pass


def safe(value):
    if isinstance(value,pd.DataFrame):return json.loads(value.to_json(orient='records',date_format='iso',double_precision=15))
    if isinstance(value,pd.Series):return safe(value.to_dict())
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


class Resources:
    def __init__(self,root):self.root=Path(root);self.lock=threading.RLock();self.cache={}
    def get(self,name):
        root=self.root
        paths={'reports':report_paths(root),'history':[root/'data/processed/full_hourly_stacked.csv',root/REPORT_DIR/'summary.json'],
               'models':model_paths(root)+[root/REPORT_DIR/'summary.json']}[name]
        key=signatures(paths)
        with self.lock:
            if self.cache.get(name,(None,))[0]!=key:
                fn={'reports':load_reports,'history':load_history,'models':load_forecaster}[name]
                try:value=fn(root)
                except Exception as exc:raise ResourceUnavailable(f'{name} unavailable: {exc}') from exc
                self.cache[name]=(key,value)
            return self.cache[name][1]

    def annotated_history(self):
        data=self.get('history');key=id(data)
        with self.lock:
            if self.cache.get('annotated',(None,))[0]!=key:
                result=data[['timestamp','demand_mw','wind_mw','solar_mw']].copy()
                flags=retrospective_target_flags(data.set_index('timestamp'))
                result['renewable_mw']=result.wind_mw+result.solar_mw
                result['net_demand_mw']=result.demand_mw-result.renewable_mw
                result['renewable_share_pct']=100*result.renewable_mw/result.demand_mw.where(result.demand_mw.gt(0))
                result['retrospective_bad']=flags.any(axis=1).to_numpy()
                names=list(flags)
                result['quality_reasons']=[json.dumps([name for name,active in zip(names,row) if active]) for row in flags.to_numpy(dtype=bool)]
                self.cache['annotated']=(key,result)
            return self.cache['annotated'][1]


def api_endpoint(fn):
    @wraps(fn)
    async def wrapped(request):
        # Starlette's thread pool keeps read/status requests responsive during inference.
        from starlette.concurrency import run_in_threadpool
        try:return await run_in_threadpool(fn,request)
        except ResourceUnavailable as exc:return JSONResponse({'error':str(exc),'code':'artifact_unavailable'},status_code=503)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            return JSONResponse({'error':str(exc),'code':'invalid_request'},status_code=422)
    return wrapped


def filters(request):
    q=request.query_params
    quantity=q.get('quantity','net');h=int(q.get('horizon',24));cohort=q.get('cohort','delivered');month=q.get('month','all')
    if quantity not in ['net',*COMPONENTS] or not 1<=h<=24 or cohort not in ['delivered','selected_only']:
        raise ValueError('Unsupported quantity, horizon or scoring cohort')
    reports=request.app.state.resources.get('reports')
    if month not in set(reports.metrics.month):raise ValueError('Unknown evaluation month')
    return reports,quantity,h,cohort,month


@api_endpoint
def status(request):
    store=request.app.state.resources;availability={};history=None;reports=None
    for name in ['reports','history','models']:
        try:
            value=store.get(name);availability[name]={'available':True,'detail':'Validated local artifacts'}
            if name=='history':history=value
            if name=='reports':reports=value
        except ResourceUnavailable as exc:availability[name]={'available':False,'detail':str(exc)}
    return JSONResponse(safe({'api':'connected','coverage':{'id':'IN-national','label':'India · national aggregate'},
        'resources':availability,'dataset':None if history is None else {'first':history.timestamp.min(),'last':history.timestamp.max(),'rows':len(history)},
        'model_id':'grid_curve_v1','horizons':list(range(1,25)),
        'months':[] if reports is None else sorted(set(reports.metrics.month)-{'all'}),
        'uptime_seconds':time.monotonic()-request.app.state.started,
        'last_successful_forecast':request.app.state.last_forecast,'optimization':'deferred',
        'training':'disabled','timestamp_convention':'timezone-naive source clock; no conversion',
        'evaluation_context':'retrospective; historical data have informed development'}))


async def forecast_request(request):
    try:body=await request.json()
    except Exception:return JSONResponse({'error':'Expected a JSON request body','code':'invalid_request'},status_code=422)
    if not isinstance(body,dict) or not isinstance(body.get('origin'),str):
        return JSONResponse({'error':'origin must be an hourly timestamp string','code':'invalid_request'},status_code=422)
    if body.get('coverage','IN-national')!='IN-national':return JSONResponse({'error':'Only national aggregate coverage is available'},status_code=422)
    try:origin=parse_origin(body['origin'])
    except ValueError as exc:return JSONResponse({'error':str(exc)},status_code=422)
    case=body.get('case','normal')
    if case not in ['normal','fallback','blocked'] or (case=='blocked' and body.get('demo') is not True):
        return JSONResponse({'error':'Synthetic blocked input requires explicit demo=true'},status_code=422)
    from starlette.concurrency import run_in_threadpool
    def execute():
        with request.app.state.inference_lock:
            store=request.app.state.resources;model=store.get('models');history=store.get('history')
            started=time.perf_counter();result=forecast(model,history,str(origin),case)
            elapsed=(time.perf_counter()-started)*1000
            request.app.state.last_forecast={'origin':str(origin),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'status':result['status'],'synthetic':case=='blocked'}
            return {'forecast':result,'table':safe(forecast_table(result)),'coverage':'IN-national',
                    'synthetic':case=='blocked','inference_ms':elapsed}
    try:return JSONResponse(safe(await run_in_threadpool(execute)))
    except ResourceUnavailable as exc:return JSONResponse({'error':str(exc),'code':'artifact_unavailable'},status_code=503)
    except (ValueError,KeyError,TypeError) as exc:return JSONResponse({'error':str(exc),'code':'invalid_request'},status_code=422)


@api_endpoint
def history(request):
    store=request.app.state.resources;data=store.annotated_history();q=request.query_params
    end=parse_origin(q.get('end',str(data.timestamp.max())))
    start=parse_origin(q.get('start',str(end-pd.Timedelta(hours=167))))
    if start>end:raise ValueError('start must precede end')
    offset=int(q.get('offset',0));limit=int(q.get('limit',5000))
    if offset<0 or not 1<=limit<=5000:raise ValueError('offset must be nonnegative; limit must be 1--5000')
    selected=data.loc[data.timestamp.between(start,end)]
    if q.get('format')=='csv':return Response(csv_bytes(selected),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="gridcast_history.csv"'})
    return JSONResponse(safe({'rows':selected.iloc[offset:offset+limit],'total':len(selected),'offset':offset,'limit':limit,
        'quality_context':'retrospective target audit; never prediction inputs','coverage':'IN-national'}))


@api_endpoint
def metrics(request):
    reports,quantity,h,cohort,month=filters(request)
    row=metric_record(reports,quantity,h,cohort,month)
    audited,counts,reasons=audit_slice(reports,h,month,cohort)
    scored=audited.loc[audited['score_'+cohort]]
    a=scored.actual_net_mw.to_numpy() if quantity=='net' else scored['actual_'+quantity+'_mw'].to_numpy()
    p=scored.net_demand_mw.to_numpy() if quantity=='net' else scored[quantity+'_mw'].to_numpy()
    b=scored.baseline_net_demand_mw.to_numpy() if quantity=='net' else scored['baseline_'+quantity+'_mw'].to_numpy()
    m=np.isfinite(a)&np.isfinite(p)&np.isfinite(b)&(np.abs(a)>0)
    mape={'forecast_pct':float(100*np.mean(np.abs((p[m]-a[m])/a[m]))) if m.any() else None,
          'baseline_pct':float(100*np.mean(np.abs((b[m]-a[m])/a[m]))) if m.any() else None,
          'paired_nonzero_actual_rows':int(m.sum()),'excluded_zero_actual_rows':int((np.abs(a)==0).sum()),
          'definition':'Derived MAPE; identical paired nonzero-actual rows; not the saved WAPE metric'}
    monthly=reports.metrics.loc[reports.metrics.quantity.eq(quantity)&reports.metrics.horizon.eq(h)&reports.metrics.cohort.eq(cohort)&reports.metrics.month.ne('all')]
    error=p-a
    bins,edges=np.histogram(error,bins=24) if len(error) else (np.array([],dtype=int),np.array([]))
    sample=scored.iloc[np.linspace(0,len(scored)-1,min(len(scored),240),dtype=int)] if len(scored) else scored
    fields=['target_timestamp','actual_net_mw','net_demand_mw'] if quantity=='net' else ['target_timestamp','actual_'+quantity+'_mw',quantity+'_mw']
    return JSONResponse(safe({'saved':row,'counts':counts,'monthly':monthly,'mape':mape,
        'error_histogram':{'counts':bins,'edges':edges,'signed_error':'forecast minus actual'},
        'comparison_sample':sample[fields],'comparison_is_display_sample':len(scored)>240,'retrospective':True}))


@api_endpoint
def audit(request):
    reports,quantity,h,cohort,month=filters(request)
    selected,counts,reasons=audit_slice(reports,h,month,cohort)
    excluded=selected.loc[~selected.trusted_target,['origin','horizon','target_timestamp','target_exclusion_reasons']]
    if request.query_params.get('format')=='csv':return Response(csv_bytes(excluded),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="gridcast_exclusions.csv"'})
    offset=int(request.query_params.get('offset',0));limit=int(request.query_params.get('limit',5000))
    if offset<0 or not 1<=limit<=5000:raise ValueError('Invalid audit pagination')
    return JSONResponse(safe({'counts':counts,'reasons':reasons,'rows':excluded.iloc[offset:offset+limit],
        'total':len(excluded),'policy':reports.summary['target_policy'],'context':'retrospective target audit; reasons can overlap'}))


@api_endpoint
def model_info(request):
    q=request.query_params;c=q.get('component','demand');h=int(q.get('horizon',24))
    if c not in COMPONENTS or not 1<=h<=24:raise ValueError('Unknown component or horizon')
    model=request.app.state.resources.get('models');key=f'{c}_h{h:02d}';entry=model.manifest['entries'][key]
    scores=model.models[key].get_booster()[:entry['best_iteration']+1].get_score(importance_type='gain')
    importance=sorted([{'feature':name,'gain':float(scores.get(name,0))} for name in model.models[key].get_booster().feature_names],key=lambda x:x['gain'],reverse=True)
    return JSONResponse(safe({'id':key,'entry':entry,'feature_importance':importance,
        'importance_label':'XGBoost split-gain importance','interpretation':'Split statistic over selected prediction trees, not a causal explanation',
        'prediction_iteration_range':[0,entry['best_iteration']+1],
        'fit_scope':model.manifest['fit_scope'],'inference_policy':model.manifest['inference_policy'],
        'source':model.manifest['source'],'versions':model.manifest['versions']}))


@api_endpoint
def logs(request):return JSONResponse({'rows':list(request.app.state.events),'context':'Actual current-session HTTP requests only; cleared on restart'})


class LoggingMiddleware:
    def __init__(self,app,owner):self.app=app;self.owner=owner
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        start=time.perf_counter();status_code=500;identifier=uuid.uuid4().hex[:12]
        async def sending(message):
            nonlocal status_code
            if message['type']=='http.response.start':
                status_code=message['status'];message['headers'].append((b'x-request-id',identifier.encode()))
            await send(message)
        try:await self.app(scope,receive,sending)
        finally:
            if scope['path'].startswith('/api/'):
                self.owner.state.events.appendleft({'id':identifier,'time_utc':datetime.now(timezone.utc).isoformat(),
                    'method':scope['method'],'path':scope['path'],'status':status_code,'duration_ms':(time.perf_counter()-start)*1000})


def create_app(root=ROOT,guards=True):
    if guards:
        from services.runtime import install_guards
        install_guards()
    root=Path(root);dist=root/'frontend/dist'
    async def spa(request):
        if request.url.path.startswith('/api/'):
            return JSONResponse({'error':'Unknown API endpoint'},status_code=404)
        if not (dist/'index.html').is_file():return JSONResponse({'error':'Build the frontend with npm run build before starting the demo'},status_code=503)
        return FileResponse(dist/'index.html')
    routes=[Route('/api/v1/status',status),Route('/api/v1/forecast',forecast_request,methods=['POST']),
        Route('/api/v1/history',history),Route('/api/v1/metrics',metrics),Route('/api/v1/audit',audit),
        Route('/api/v1/models',model_info),Route('/api/v1/logs',logs)]
    if dist.exists():routes.append(Mount('/assets',app=StaticFiles(directory=dist/'assets',check_dir=False)))
    # Public geometry has a stable URL in both the Vite dev server and production.
    async def geometry(request):return FileResponse(root/'frontend/public/india.geojson',media_type='application/geo+json')
    routes.extend([Route('/india.geojson',geometry),Route('/{path:path}',spa)])
    app=Starlette(routes=routes)
    app.state.resources=Resources(root);app.state.events=deque(maxlen=200);app.state.started=time.monotonic()
    app.state.last_forecast=None;app.state.inference_lock=threading.Lock()
    app.add_middleware(LoggingMiddleware,owner=app)
    return app
