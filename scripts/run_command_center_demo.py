"""Reproduce the command-center API walkthrough with all fitting/dispatch disabled."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from starlette.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from command_api.app import create_app,ResourceUnavailable


def main():
    parent=ROOT/'reports/demos/command_center_v1';parent.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    app=create_app(ROOT);client=TestClient(app);checks={};cases={}
    for name,origin in [('normal','2025-06-24 23:00:00'),('fallback','2025-01-09 15:00:00'),('blocked','2025-06-24 23:00:00')]:
        result=client.post('/api/v1/forecast',json={'origin':origin,'case':name,'demo':name=='blocked'})
        assert result.status_code==200,result.text
        value=result.json();assert len(value['forecast']['points'])==24
        assert value['forecast']['target_quality']=='unknown' and 'simulation' not in value['forecast']
        if name=='normal':assert value['forecast']['complete_curve']
        if name=='fallback':assert any(p['components']['demand']['status']=='fallback' for p in value['forecast']['points'])
        if name=='blocked':assert all(r['wind_mw'] is None and r['net_demand_mw'] is None for r in value['table'])
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        cases[name]={'origin':origin,'status':value['forecast']['status'],'synthetic':value['synthetic']}
    cli=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'src/forecasting/predict_grid_curve.py'),
        '--origin','2025-06-24 23:00:00','--history',str(ROOT/'data/processed/full_hourly_stacked.csv')],check=True,capture_output=True,text=True)
    expected=json.loads(cli.stdout)
    assert expected==json.loads((output/'normal.json').read_text())['forecast']
    checks['api_matches_existing_cli_exactly']=True
    for name,url in [('metrics','/api/v1/metrics'),('solar_metrics','/api/v1/metrics?quantity=solar&horizon=24'),
        ('model','/api/v1/models?component=demand&horizon=24'),('audit','/api/v1/audit?month=2025-04'),
        ('status','/api/v1/status')]:
        response=client.get(url);assert response.status_code==200,response.text
        (output/(name+'.json')).write_text(json.dumps(response.json(),indent=2,allow_nan=False)+'\n')
    history=client.get('/api/v1/history?start=2025-04-04%2000:00:00&end=2025-04-04%2023:00:00&format=csv')
    assert history.status_code==200;(output/'april4_history.csv').write_text(history.text)
    original=app.state.resources.get
    def outage(name):
        if name=='models':raise ResourceUnavailable('Browser/demo model-outage fixture; real artifacts untouched')
        return original(name)
    with patch.object(app.state.resources,'get',side_effect=outage):
        assert client.post('/api/v1/forecast',json={'origin':'2025-06-24 23:00:00'}).status_code==503
        assert client.get('/api/v1/metrics').status_code==200
    checks.update(normal_fallback_blocked=True,paired_metrics_and_mape=True,selected_tree_gain=True,
        model_outage_preserves_history=True,training_disabled=True,battery_disabled=True)
    receipt={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,'checks':checks,
        'context':'retrospective national aggregate demonstration; synthetic cases labelled; optimization deferred',
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'passed','output':str(output),'checks':checks},indent=2))


if __name__=='__main__':main()
