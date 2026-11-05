"""Load-only historical verification. All training entry points are disabled."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.demand_model import load_demand_bundle, sha256
from common.features import build_features
from common.splits import chronological_split

ROOT = Path(__file__).resolve().parents[2]


def verify(root, bundle):
    predictor = load_demand_bundle(bundle)
    meta = predictor.metadata
    data = root / meta['source_data']['path']
    if sha256(data) != meta['source_data']['sha256']:
        raise ValueError('Source data differs from recorded training data')
    for relative, expected in meta['code_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Training/feature code changed: {relative}')
    frame = build_features(pd.read_csv(data, parse_dates=['Timestamp']))
    names = [f['name'] for f in meta['features']]
    usable = frame.dropna(subset=names + ['Target']).reset_index(drop=True)
    train, validation, test = chronological_split(usable)
    for name, split in [('train',train), ('validation',validation), ('test',test)]:
        observed = {'rows':len(split), 'first_origin':str(split.Timestamp.min()),
                    'last_origin':str(split.Timestamp.max()),
                    'first_target':str(split['Target Timestamp'].min()),
                    'last_target':str(split['Target Timestamp'].max())}
        if observed != meta['splits'][name]:
            raise ValueError(f'Reconstructed {name} dates/count differ from metadata')
    evidence = json.loads((bundle / 'training_reference.json').read_text())
    if evidence['model_sha256'] != meta['model_sha256'] or evidence['test_reference_sha256'] != sha256(bundle/'test_reference.csv'):
        raise ValueError('Training reference does not match bundle')
    reference = pd.read_csv(bundle/'test_reference.csv')
    saved = pd.read_csv(root/'reports/experiments/xgboost_demand_predictions.csv')
    for label, table in [('full precision reference',reference), ('experiment CSV',saved)]:
        if len(table) != len(test) or table[['Forecast Origin','Target Timestamp']].duplicated().any():
            raise ValueError(f'{label}: incomplete/duplicate alignment')
        for col, times in [('Forecast Origin',test.Timestamp),('Target Timestamp',test['Target Timestamp'])]:
            if not pd.to_datetime(table[col]).reset_index(drop=True).equals(times.reset_index(drop=True)):
                raise ValueError(f'{label}: timestamp mismatch')
        np.testing.assert_allclose(table['Actual Demand (MW)'], test.Target, rtol=0, atol=1e-9)
    # This is the only model prediction path, in a process where fit/train raise.
    loaded = predictor.predict(test[names])
    np.testing.assert_allclose(loaded,reference['XGBoost Forecast (MW)'],rtol=0,atol=1e-6)
    np.testing.assert_allclose(loaded,saved['XGBoost Forecast (MW)'],rtol=0,atol=.02)
    actual = test.Target.to_numpy()
    errors = actual-loaded
    metrics = {'MAE (MW)':float(np.abs(errors).mean()),
               'RMSE (MW)':float(np.sqrt(np.mean(errors**2))),
               'MAPE (%)':float(np.mean(np.abs(errors)/actual)*100)}
    csv_metrics = pd.read_csv(root/'reports/experiments/xgboost_demand_metrics.csv').set_index('Model').loc['XGBoost']
    for key,value in metrics.items():
        np.testing.assert_allclose(value,meta['test_metrics'][key],rtol=0,atol=1e-9)
        np.testing.assert_allclose(value,csv_metrics[key],rtol=0,atol=1e-9)
    target = pd.Timestamp('2025-01-10 15:00:00')
    diagnostic = frame.loc[frame.Timestamp.eq(target-pd.Timedelta(hours=24))]
    if len(diagnostic)!=1 or diagnostic.quality_eligible.any() or test.Timestamp.eq(diagnostic.Timestamp.iloc[0]).any():
        raise ValueError('January spot-check should be excluded from scoring')
    return {'status':'passed', 'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'model_sha256':meta['model_sha256'], 'training_disabled':True,
            'rows':len(test), 'one_to_one_timestamp_alignment':True,
            'max_in_memory_reference_difference_mw':float(np.max(np.abs(loaded-reference['XGBoost Forecast (MW)']))),
            'max_existing_csv_difference_mw':float(np.max(np.abs(loaded-saved['XGBoost Forecast (MW)']))),
            'test_metrics':metrics,'preexisting_experiment_check':evidence['existing_experiment_comparison'],
            'january_10':{'target':str(target),'included_in_scoring':False,
                          'label':'diagnostic only; frozen demand in input history',
                          'prediction_mw':float(predictor.predict(diagnostic[names])[0])}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    bundle=args.bundle or args.root/'models/demand_v1'
    with patch.object(xgboost.XGBRegressor,'fit',side_effect=AssertionError('Training is forbidden in load-only verification')), \
         patch.object(xgboost,'train',side_effect=AssertionError('Training is forbidden in load-only verification')), \
         patch.object(xgboost.Booster,'update',side_effect=AssertionError('Training is forbidden in load-only verification')):
        report=verify(args.root,bundle)
    rendered=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__=='__main__':
    main()
