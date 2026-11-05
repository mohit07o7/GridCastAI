"""Serve the production React build and verified local API on loopback only."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8502)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Use a port between 1024 and 65535')
    try:
        import uvicorn
        from command_api.app import create_app
    except ImportError as exc:
        print(f'Missing API dependency: {exc}. Install the documented Python requirements first.',file=sys.stderr);return 1
    if not (ROOT/'frontend/dist/index.html').is_file():
        print('Missing frontend build. Run: cd frontend && npm ci && npm run build',file=sys.stderr);return 1
    app=create_app(ROOT)
    checks={}
    for name in ['reports','history','models']:
        try:app.state.resources.get(name);checks[name]='available'
        except Exception as exc:checks[name]=str(exc)
    print(json.dumps({'address':'127.0.0.1','port':args.port,'resources':checks,'training':'disabled','optimization':'deferred'},indent=2),flush=True)
    if args.check:return 0 if all(v=='available' for v in checks.values()) else 1
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')
    return 0


if __name__=='__main__':sys.exit(main())
