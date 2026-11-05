"""Load a saved 1--24 hour forecast curve and optional illustrative battery plan."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.curve_model import GridCurveForecaster
from common.battery_dispatch import load_scenario,optimize_dispatch

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin',required=True)
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--scenario')
    parser.add_argument('--soc-mwh',type=float)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        if args.soc_mwh is not None and not args.scenario:raise ValueError('--soc-mwh requires --scenario')
        if args.output and args.output.exists():raise ValueError('Output already exists; refuse to overwrite any existing file')
        service=GridCurveForecaster(args.root)
        result=service.forecast(pd.read_csv(args.history),args.origin)
        if args.scenario:
            scenario=load_scenario(args.scenario,args.root)
            plan=optimize_dispatch(result,scenario,args.soc_mwh)
            if isinstance(plan['hours'],pd.DataFrame):
                hours=plan['hours'].copy()
                hours['target_timestamp']=hours.target_timestamp.astype(str)
                plan['hours']=hours.to_dict(orient='records')
            result['simulation']={'label':'illustrative scaled simulation; not operational dispatch',
                'power_to_energy_assumption':'each hourly MW reading held constant for one hour',
                'parameters':asdict(scenario),
                'initial_soc_mwh':scenario.initial_soc_mwh if args.soc_mwh is None else args.soc_mwh,
                'plan':plan}
        rendered=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if args.output:args.output.write_text(rendered)
        print(rendered,end='')
        return 2 if result['status']=='blocked' else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'error','error':str(exc)}),file=sys.stderr)
        return 1


if __name__=='__main__':sys.exit(main())
