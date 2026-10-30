"""Trace a target and replay the saved demand experiment without overwriting it.

Example: python src/investigations/reproduce_demand_spot_check.py --retrain \
    --compare-root /Users/mohityadav/Desktop/GridCastAI \
    --raw-workbook '/Users/mohityadav/Desktop/GridCastAI/data/raw/January 2024- June 2025.xlsx'

Uses the experiment's actual source up to model.fit (AST boundary), never its
CSV-writing evaluation section. Only use trusted project roots. Without
--retrain, stops before model construction and checks data/features/saved rows.
Use --historical for the preserved original Desktop experiment. Current-policy
excluded targets are reported as diagnostic only, never scored.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(frame, column, timestamp):
    rows = frame.loc[pd.to_datetime(frame[column]) == timestamp]
    if len(rows) != 1:
        raise ValueError(f'Expected one {column}={timestamp}, found {len(rows)}')
    return rows.iloc[0]


def trace(root, target, retrain, historical=False):
    script = root / 'src/forecasting/xgboost_demand_forecast.py'
    data = root / 'data/processed/demand_model_data.csv'
    saved_path = root / 'reports/experiments/xgboost_demand_predictions.csv'
    if historical:
        base = root / 'reports/historical/pre_shared_2026_10_31'
        source = base / 'src/forecasting/xgboost_demand_forecast.py'
        saved_path = base / 'reports/experiments/xgboost_demand_predictions.csv'
    else:
        source = script
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in node.targets) and not retrain:
            break
        selected.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'model' and node.value.func.attr == 'fit'):
            break
    else:
        raise ValueError('Cannot locate safe model.fit boundary')
    ns = {'__file__': str(script), '__name__': '__spot_check__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), 'exec'), ns)
    origin = target - pd.Timedelta(hours=24)
    df, features = ns['df'], ns['feature_cols']
    row = unique(ns['model_df'], 'Timestamp', origin)
    scored = bool(ns['test']['Timestamp'].eq(origin).any())
    saved = pd.read_csv(saved_path)
    recorded = unique(saved, 'Target Timestamp', target) if scored else pd.Series({'Forecast Origin': str(origin), 'Target Timestamp': str(target), 'Actual Demand (MW)': row['Target'], 'Seasonal Naive Forecast (MW)': row['demand_mw']})
    assert pd.Timestamp(recorded['Forecast Origin']) == origin
    actual = unique(df, 'Timestamp', target)['demand_mw']
    baseline = unique(df, 'Timestamp', origin)['demand_mw']
    assert actual == row['Target'] == recorded['Actual Demand (MW)']
    assert baseline == recorded['Seasonal Naive Forecast (MW)']
    stacked = pd.read_csv(root / 'data/processed/full_hourly_stacked.csv')
    for stamp in [origin, target]:
        a, b = unique(stacked, 'timestamp', stamp), unique(df, 'Timestamp', stamp)
        for col in ['demand_mw', 'wind_mw', 'solar_mw']:
            assert a[col] == b[col]
    provenance = {}
    for name in features:
        if '_lag_' in name:
            hours = int(name.rsplit('_', 1)[1])
            provenance[name] = str(origin - pd.Timedelta(hours=hours))
        elif '_mean_' in name:
            hours = int(name.rsplit('_', 1)[1][:-1])
            provenance[name] = [str(origin - pd.Timedelta(hours=hours)), str(origin - pd.Timedelta(hours=1))]
        else:
            provenance[name] = str(origin)
    result = {
        'root': str(root), 'sha256': {str(p.relative_to(root)): digest(p) for p in [source, data, saved_path]},
        'origin': str(origin), 'target': str(target), 'historical': historical, 'included_in_scoring': scored, 'quality_input_bad': bool(row.get('quality_input_bad', False)), 'quality_target_bad': bool(row.get('quality_target_bad', False)), 'data_rows': len(df),
        'usable_rows': len(ns['model_df']), 'saved_row': recorded.to_dict(),
        'features': {f: float(row[f]) for f in features}, 'feature_source_timestamps': provenance,
        'splits': {key: {'rows': len(ns[key]), 'first_origin': str(ns[key]['Timestamp'].iloc[0]),
                        'last_origin': str(ns[key]['Timestamp'].iloc[-1]),
                        'last_target': str(ns[key]['Timestamp'].iloc[-1] + pd.Timedelta(hours=24))}
                   for key in ['train', 'validation', 'test']},
    }
    if retrain:
        model = ns['model']
        predictions = model.predict(ns['X_test'])
        point = float(model.predict(row[features].astype(float).to_frame().T)[0])
        aligned = saved.merge(pd.DataFrame({'Forecast Origin': ns['test']['Timestamp'].astype(str), 'replayed': predictions}), on='Forecast Origin', validate='one_to_one')
        assert len(aligned) == len(saved) == len(predictions)
        delta = np.abs(aligned['replayed'] - aligned['XGBoost Forecast (MW)'])
        result['replay'] = {'prediction': point, 'prediction_is_diagnostic_only': not scored, 'best_iteration': model.best_iteration,
                            'parameters': {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in model.get_params().items()}, 'max_saved_csv_difference_mw': float(delta.max()),
                            'all_rows_match_within_0_02_mw': bool((delta <= 0.02).all())}
        if not result['replay']['all_rows_match_within_0_02_mw']:
            raise AssertionError(f'Replay differs from saved run: max {delta.max()} MW')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--compare-root', type=Path)
    parser.add_argument('--target', default='2025-01-10 15:00:00')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--historical', action='store_true', help='Replay preserved pre-shared Desktop run')
    parser.add_argument('--raw-workbook', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = pd.Timestamp(args.target)
    if target.tzinfo is not None or target != target.floor('h'):
        parser.error('Use an exact timezone-naive source hour; no timezone conversion is applied.')
    roots = [args.root.resolve()] + ([args.compare_root.resolve()] if args.compare_root else [])
    result = {'versions': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__},
              'runs': [trace(root, target, args.retrain, args.historical) for root in roots]}
    if args.raw_workbook:
        raw = pd.read_excel(args.raw_workbook, sheet_name='Report')
        raw['Timestamp'] = pd.to_datetime(raw['Timestamp'], dayfirst=True, errors='coerce')
        result['raw_workbook'] = {'path': str(args.raw_workbook), 'sha256': digest(args.raw_workbook), 'rows': []}
        for stamp in [target - pd.Timedelta(hours=24), target]:
            row = unique(raw, 'Timestamp', stamp)
            result['raw_workbook']['rows'].append(row.to_dict())
            for root in roots:
                data = pd.read_csv(root / 'data/processed/demand_model_data.csv')
                processed = unique(data, 'Timestamp', stamp)
                for raw_col, col in [('Demand (MW)', 'demand_mw'), ('Wind (MW)', 'wind_mw'), ('Solar (MW)', 'solar_mw')]:
                    assert processed[col] == (max(0, row[raw_col]) if col == 'solar_mw' else row[raw_col])
    rendered = json.dumps(result, indent=2, default=str, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
