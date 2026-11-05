"""Predict demand, wind and solar for one target, 24 hours after origin."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common.grid_prediction import GridForecaster, ROOT
from common.inference_history import HistoryError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', required=True, help='Exact timezone-naive source hour')
    parser.add_argument('--history', required=True, type=Path, help='CSV with timestamp, demand_mw, wind_mw, solar_mw')
    parser.add_argument('--output', type=Path, help='Optional JSON file; also print JSON to stdout')
    parser.add_argument('--root', type=Path, default=ROOT, help='Project containing saved artifacts')
    args = parser.parse_args()
    try:
        result = GridForecaster(args.root).forecast(pd.read_csv(args.history), args.origin)
        rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            # Never let an output flag overwrite input history or a saved artifact.
            output = args.output.resolve()
            protected = [args.history.resolve(), *(args.root / 'models').resolve().rglob('*')]
            if output in protected:
                raise HistoryError('Output must not overwrite input history or model artifacts')
            output.write_text(rendered)
        sys.stdout.write(rendered)
        return 2 if result['status'] == 'blocked' else 0
    except (HistoryError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
