"""Preflight local artifacts and start Streamlit without installation or training."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Port must be between 1024 and 65535')
    missing=[n for n in ['streamlit','plotly','pandas','xgboost'] if importlib.util.find_spec(n) is None]
    if missing:
        print('Missing dashboard packages: '+', '.join(missing)+'\nInstall once: .venv/bin/python -m pip install -r requirements-dashboard.txt',file=sys.stderr)
        return 1
    from dashboard.runtime import install_guards
    install_guards()
    from dashboard.service import load_reports,load_history,load_forecaster
    checks={}
    for name,fn in [('reports',load_reports),('history',load_history),('models',load_forecaster)]:
        try:fn(ROOT);checks[name]='available'
        except Exception as exc:checks[name]='unavailable: '+str(exc)
    print(json.dumps({'startup':checks,'training':'disabled','battery':'deferred and disabled','address':'127.0.0.1','port':args.port},indent=2),flush=True)
    if args.check:return 0 if all(v=='available' for v in checks.values()) else 1
    # Degraded startup is deliberate: historical reports survive model outages.
    from streamlit.web import cli
    sys.argv=['streamlit','run',str(ROOT/'src/dashboard/app.py'),'--server.address','127.0.0.1',
        '--server.port',str(args.port),'--server.headless','true','--browser.gatherUsageStats','false',
        '--theme.base','light','--theme.primaryColor','#087f8c','--client.toolbarMode','minimal']
    return cli.main()


if __name__=='__main__':sys.exit(main())
