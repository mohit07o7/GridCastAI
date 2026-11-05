"""Train 72 direct curve candidates, select on validation only, save a new bundle."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.curve_features import (BASELINES, COMPONENTS, FEATURES, FEATURE_VERSION,
    POLICY, HOUR, TEST_START, VALIDATION_START, CurveHistory, split_masks)
from common.curve_model import METHOD, digest, fingerprint, json_write, predict_candidate
from common.grid_evaluation import retrospective_target_flags

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = dict(n_estimators=1000, early_stopping_rounds=50, max_depth=4,
    learning_rate=.03, subsample=.8, colsample_bytree=.8, min_child_weight=5,
    reg_lambda=2., objective='reg:absoluteerror', eval_metric='mae',
    random_state=42, n_jobs=4, tree_method='hist')


def target_bad_flags(context, component):
    """Retrospective labels ONLY; never used to construct a forecast feature."""
    flags = retrospective_target_flags(context.data)
    value = context.data[f'{component}_mw']
    bad = ~np.isfinite(value) | value.lt(0) | flags.target_all_zero
    if component == 'demand':
        bad |= flags.target_nonpositive_demand | flags.target_april_anomaly | flags.target_frozen_demand
    elif component == 'wind':
        bad |= flags.target_frozen_wind | flags.target_renewable_inconsistency
    else:
        bad |= flags.target_positive_solar_flatline | flags.target_renewable_inconsistency
    return bad


def date_range(origins, horizon):
    return {'rows': len(origins), 'first_origin': str(origins.min()), 'last_origin': str(origins.max()),
            'first_target': str((origins+horizon*HOUR).min()),
            'last_target': str((origins+horizon*HOUR).max())}


def train(root):
    source = root/'data/processed/full_hourly_stacked.csv'
    destination = root/'models/grid_curve_v1'
    if destination.exists():
        raise ValueError('grid_curve_v1 already exists; preserve it and choose a new version deliberately')
    raw = pd.read_csv(source)
    context = CurveHistory(raw)
    origins = context.index[168:-24]
    nominal = split_masks(origins)
    meta = {'schema_version': 1, 'model_id': 'grid_curve_v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'features': [{'name': x, 'dtype': 'float64'} for x in FEATURES],
            'feature_version': FEATURE_VERSION, 'inference_policy': POLICY,
            'training_target_policy': 'curve_component_target_quality_v1',
            'evaluation_target_policy': 'grid_evaluation_target_v1',
            'horizons': list(range(1,25)), 'parameters': PARAMETERS,
            'fit_scope': 'training only; validation early stopping and method selection; no refit',
            'validation_start_origin': str(VALIDATION_START), 'test_start_origin': str(TEST_START),
            'max_horizon_purge_hours': 24,
            'selection': 'lowest validation MAE on identical eligible rows; ties within 1e-9 MW favor baseline',
            'baseline_tie_order': list(BASELINES),
            'source': {'path': str(source.relative_to(root)), 'sha256': digest(source)},
            'code_sha256': {str(p.relative_to(root)): digest(p) for p in [
                root/'src/common/curve_features.py', root/'src/common/curve_model.py', Path(__file__)]},
            'versions': {'python': platform.python_version(), **{n: version(n) for n in
                         ['numpy','pandas','xgboost','scikit-learn','scipy']}},
            'entries': {}, 'run_context': 'retrospective; historical test has informed development'}
    references = []
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='grid-curve-', dir=destination.parent) as tmp:
        staging = Path(tmp)
        for c in COMPONENTS:
            model_quality = ~context.quality(c, origins).any(axis=1).to_numpy()
            bad_target = target_bad_flags(context, c)
            for h in range(1,25):
                key = f'{c}_h{h:02d}'
                features = context.features(c,h,origins)
                actual = context.data[f'{c}_mw'].reindex(origins+h*HOUR).to_numpy(dtype=float)
                eligible = model_quality & ~bad_target.reindex(origins+h*HOUR, fill_value=True).to_numpy()
                eligible &= np.isfinite(actual) & np.isfinite(features.to_numpy()).all(axis=1)
                masks = {s: nominal[s] & eligible for s in nominal}
                if min(int(m.sum()) for m in masks.values()) < 10:
                    raise ValueError(f'Insufficient eligible data for {key}')
                model = XGBRegressor(**PARAMETERS)
                model.fit(features.loc[masks['train']],
                          actual[masks['train']]-features.loc[masks['train'],'target_previous_day'].to_numpy(),
                          eval_set=[(features.loc[masks['validation']],
                            actual[masks['validation']]-features.loc[masks['validation'],'target_previous_day'].to_numpy())],
                          verbose=False)
                entry = {'component': c, 'horizon': h, 'model_file': key+'.json',
                         'best_iteration': int(model.best_iteration),
                         'boosted_rounds': model.get_booster().num_boosted_rounds(),
                         'splits': {s: date_range(origins[m],h) for s,m in masks.items()}}
                validation = masks['validation']
                predictions = predict_candidate(model,features.loc[validation],entry)
                scores = {METHOD: float(np.abs(predictions-actual[validation]).mean())}
                for method in BASELINES:
                    scores[method] = float(np.abs(context.baseline(c,method,h,origins[validation])-actual[validation]).mean())
                ranking = sorted(BASELINES,key=lambda m:(scores[m],BASELINES.index(m)))
                chosen = METHOD if scores[METHOD] < scores[ranking[0]]-1e-9 else ranking[0]
                entry.update(validation_mae_mw=scores, baseline_ranking=ranking,
                             selected_method=chosen,
                             available_after_target=entry['splits']['validation']['last_target'])
                model.get_booster().set_attr(gridcast_curve_metadata_sha256=fingerprint(entry))
                model.save_model(staging/entry['model_file'])
                entry['model_sha256'] = digest(staging/entry['model_file'])
                meta['entries'][key] = entry
                test_origins = origins[nominal['test']]
                test_features = features.loc[nominal['test']]
                candidate = predict_candidate(model,test_features,entry)
                reference = pd.DataFrame({'origin':test_origins,'target_timestamp':test_origins+h*HOUR,
                    'component':c,'horizon':h,'actual':actual[nominal['test']],
                    'candidate_mw':candidate,'eligible':eligible[nominal['test']]})
                for method in BASELINES:
                    reference[method] = context.baseline(c,method,h,test_origins)
                reference['selected_mw'] = candidate if chosen==METHOD else reference[chosen]
                references.append(reference)
                print(f'{key}: train={masks["train"].sum()} val={validation.sum()} '
                      f'best={entry["best_iteration"]} selected={chosen}',flush=True)
        reference = pd.concat(references,ignore_index=True)
        reference.to_csv(staging/'test_reference.csv',index=False,float_format='%.17g')
        meta['test_reference'] = {'path':'test_reference.csv','sha256':digest(staging/'test_reference.csv'),
                                  'rows':len(reference)}
        json_write(staging/'metadata.json',meta)
        from common.curve_model import GridCurveForecaster
        GridCurveForecaster(root,bundle=staging,strict=True)
        shutil.move(str(staging),destination)
    print(json.dumps({'bundle':str(destination),'candidates':len(meta['entries']),
        'selected_counts':{c:sum(e['selected_method']==METHOD for e in meta['entries'].values() if e['component']==c) for c in COMPONENTS},
        'test_origins':int(nominal['test'].sum())},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    train(parser.parse_args().root)
