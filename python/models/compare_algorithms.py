"""Chronological model/loss comparison with fixed previously selected features."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.experiment_features import paired_date_interval
from models.full_training_review import (
    STATS, TRAIN, TEST, KEYS, assess, calibration_fit, development_folds,
    prepare_features,
)


def candidates():
    result = [{'id': 'linear', 'family': 'linear', 'params': {}},
              {'id': 'xgb_current', 'family': 'xgb_rmse',
               'params': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': .05,
                          'subsample': .8, 'colsample_bytree': .8}}]
    shapes = [(2, 600, .03, 10, 10), (3, 600, .03, 10, 10),
              (4, 600, .03, 10, 10), (3, 1000, .02, 20, 20)]
    for loss, objective in [('rmse', 'reg:squarederror'), ('mae', 'reg:absoluteerror')]:
        for i, (depth, trees, rate, child, penalty) in enumerate(shapes):
            result.append({'id': f'xgb_{loss}_{i}', 'family': f'xgb_{loss}',
                           'params': {'objective': objective, 'n_estimators': trees,
                                      'max_depth': depth, 'learning_rate': rate,
                                      'min_child_weight': child, 'reg_lambda': penalty,
                                      'subsample': .8, 'colsample_bytree': .9}})
    for loss in ['RMSE', 'MAE']:
        for depth in [4, 6, 8]:
            result.append({'id': f'cat_{loss.lower()}_{depth}', 'family': f'cat_{loss.lower()}',
                           'params': {'loss_function': loss, 'iterations': 600,
                                      'depth': depth, 'learning_rate': .04, 'l2_leaf_reg': 10}})
    return result


def estimator(candidate):
    if candidate['id'] == 'linear':
        return LinearRegression()
    if candidate['id'].startswith('xgb'):
        return XGBRegressor(**candidate['params'], random_state=42, n_jobs=1)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**candidate['params'], random_seed=42, thread_count=1,
                             verbose=False, allow_writing_files=False)


def choose_candidate(rows, ids=None):
    """Equal-weight relative MAE across expanding development folds only."""
    frame = pd.DataFrame(rows)
    if set(frame.season) != set(TRAIN[1:]):
        raise ValueError('Selection requires precisely the two development seasons')
    baseline = frame[frame.candidate.eq('linear')].set_index('season').mae
    choices = []
    for candidate, group in frame.groupby('candidate', sort=False):
        if ids is not None and candidate not in ids:
            continue
        if set(group.season) != set(baseline.index) or len(group) != len(baseline):
            raise ValueError('Incomplete or duplicate development folds')
        score = float((group.set_index('season').mae / baseline).mean())
        choices.append((score, candidate))
    if not choices:
        raise ValueError('No candidates')
    return min(choices)[1]


def blend_prediction(linear, tree, tree_weight):
    if not 0 <= tree_weight <= 1:
        raise ValueError('Blend weight outside [0,1]')
    linear, tree = np.asarray(linear), np.asarray(tree)
    if linear.shape != tree.shape:
        raise ValueError('Blend prediction shapes differ')
    return (1 - tree_weight) * linear + tree_weight * tree


def calibrating_rows(validation, prediction, stat, training_last_date):
    output = validation[KEYS].copy()
    output['actual'], output['prediction'] = validation[stat].to_numpy(), prediction
    output['line'], output['scale'] = validation[f'{stat}_avg_season'].to_numpy(), validation[f'{stat}_scale'].to_numpy()
    output['training_last_date'] = str(training_last_date.date())
    return output


def run(source, folder):
    snapshot_file = source/'source_snapshot.json'
    feature_file = source/'frozen_features.json'
    grid = candidates()
    manifest = json.loads(feature_file.read_text())
    provenance = {'source_snapshot': str(snapshot_file),
                  'source_sha256': hashlib.sha256(snapshot_file.read_bytes()).hexdigest(),
                  'frozen_features_sha256': hashlib.sha256(feature_file.read_bytes()).hexdigest(),
                  'candidate_grid': grid, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'selection_rule': 'Equal mean development MAE relative to linear over the two expanding season folds. MAE is primary; no test outcomes tune settings or blends.'}
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'design.json').write_text(json.dumps(provenance, indent=2))
    eligible = prepare_features(json.loads(snapshot_file.read_text()))
    frames, predictions, development, selected, blends = {}, {}, [], {}, {}
    grid_by_id = {c['id']: c for c in grid}
    families = ['xgb_rmse', 'xgb_mae', 'cat_rmse', 'cat_mae']
    with threadpool_limits(limits=1):
        for stat in STATS:
            inputs = manifest['selections'][stat]['features']
            frame = eligible.dropna(subset=inputs + [stat, f'{stat}_scale'])
            if not np.isfinite(frame[inputs].to_numpy(float)).all():
                raise ValueError('Nonfinite inputs')
            frames[stat] = frame
            stat_rows = []
            for candidate in grid:
                started = time.monotonic()
                for season, train, validation in development_folds(frame):
                    model = estimator(candidate).fit(train[inputs], train[stat])
                    prediction = model.predict(validation[inputs])
                    predictions[(stat, candidate['id'], season)] = prediction
                    row = {'stat': stat, 'candidate': candidate['id'], 'family': candidate['family'],
                           'season': season, 'mae': float(np.mean(abs(validation[stat].to_numpy() - prediction))),
                           'train_rows': len(train), 'validation_rows': len(validation)}
                    stat_rows.append(row)
                    development.append(row)
                pd.DataFrame(development).to_csv(folder/'development_results.csv', index=False)
                scores = [r['mae'] for r in stat_rows if r['candidate'] == candidate['id']]
                print(f'{stat} development {candidate["id"]}: '+', '.join(f'{x:.4f}' for x in scores)+f' ({time.monotonic()-started:.1f}s)', flush=True)
            selected[stat] = {'linear': 'linear', 'xgb_current': 'xgb_current'}
            for family in families:
                ids = [c['id'] for c in grid if c['family'] == family]
                selected[stat][family] = choose_candidate(stat_rows, ids)
            # Blend weights and tree family are chosen exclusively on development rows.
            blend_rows = []
            for candidate_id in set(selected[stat][family] for family in families):
                for weight in [0., .25, .5, .75, 1.]:
                    label = f'blend__{candidate_id}__{weight}'
                    for season, _, validation in development_folds(frame):
                        prediction = blend_prediction(predictions[(stat, 'linear', season)], predictions[(stat, candidate_id, season)], weight)
                        predictions[(stat, label, season)] = prediction
                        blend_rows.append({'stat': stat, 'candidate': label, 'family': 'blend', 'season': season,
                                           'mae': float(np.mean(abs(validation[stat].to_numpy() - prediction)))})
            blend_id = choose_candidate(stat_rows + blend_rows, [r['candidate'] for r in blend_rows])
            _, tree_id, weight = blend_id.split('__')
            blends[stat] = {'id': blend_id, 'tree_id': tree_id, 'tree_weight': float(weight)}
            selected[stat]['blend'] = blend_id
            development.extend(blend_rows)
            ids = list(selected[stat].values())
            selected[stat]['primary_development_choice'] = choose_candidate(stat_rows + blend_rows, ids)
            print(f'Frozen {stat}: {selected[stat]}', flush=True)
        # All choices are frozen before evaluating any candidate on 2025-26.
        frozen = {'selected_before_test_predictions': True, 'selected': selected, 'blends': blends,
                  'features': manifest['selections'], 'training_seasons': TRAIN, 'test_season': TEST}
        (folder/'frozen_comparison.json').write_text(json.dumps(frozen, indent=2))
        pd.DataFrame(development).to_csv(folder/'development_results.csv', index=False)
        results, monthly, coverage = [], [], []
        for stat in STATS:
            frame, inputs = frames[stat], manifest['selections'][stat]['features']
            training, test = frame[frame.season.isin(TRAIN)], frame[frame.season.eq(TEST)]
            if training.date.max() >= test.date.min():
                raise ValueError('Final train/test chronology failed')
            prior = pd.read_csv(source/f'predictions_{stat}_linear_optimized.csv')
            newkeys = test[KEYS].copy()
            newkeys['date'] = newkeys.date.dt.strftime('%Y-%m-%d')
            if not prior[KEYS].reset_index(drop=True).equals(newkeys.reset_index(drop=True)):
                raise ValueError('Prediction cohort differs from previous experiment')
            coverage.append({'stat': stat, 'train_rows': len(training), 'test_rows': len(test)})
            fitted, forecasts, error_reference = {}, {}, None
            test_predictions = {}
            for label in ['linear', 'xgb_current'] + families + ['blend']:
                candidate_id = selected[stat][label]
                if label == 'blend':
                    choice = blends[stat]
                    prediction = blend_prediction(forecasts['linear'], forecasts[choice['tree_id']], choice['tree_weight'])
                    model_artifact = {'linear': fitted['linear'], 'tree': fitted[choice['tree_id']], 'tree_weight': choice['tree_weight']}
                else:
                    if candidate_id not in fitted:
                        fitted[candidate_id] = estimator(grid_by_id[candidate_id]).fit(training[inputs], training[stat])
                        forecasts[candidate_id] = fitted[candidate_id].predict(test[inputs])
                    prediction, model_artifact = forecasts[candidate_id], fitted[candidate_id]
                cal_parts = [calibrating_rows(va, predictions[(stat, candidate_id, season)], stat, tr.date.max())
                             for season, tr, va in development_folds(frame)]
                calibration = pd.concat(cal_parts, ignore_index=True)
                classifier, quantiles, cal_rate = calibration_fit(calibration)
                row, output = assess(test, prediction, stat, classifier, quantiles, cal_rate)
                errors = abs(test[stat].to_numpy() - prediction)
                if label == 'linear':
                    error_reference = errors
                    if not np.allclose(prior.prediction, prediction, atol=1e-10, rtol=1e-10):
                        raise ValueError('Linear reference does not reproduce prior run')
                row.update({'stat': stat, 'model': label, 'candidate_id': candidate_id,
                            'primary_development_choice': candidate_id == selected[stat]['primary_development_choice'],
                            'train_rows': len(training), 'test_rows': len(test), 'calibration_rows': len(calibration),
                            'mae_change_vs_linear': float(np.mean(errors-error_reference)),
                            'paired_95pct_vs_linear': paired_date_interval(test.date, error_reference, errors)})
                output.to_csv(folder/f'predictions_{stat}_{label}.csv', index=False)
                calibration.to_csv(folder/f'calibration_{stat}_{label}.csv', index=False)
                joblib.dump({'model': model_artifact, 'model_label': label, 'candidate_id': candidate_id,
                             'features': inputs, 'direction_calibrator': classifier, 'interval_quantiles': quantiles,
                             'training_seasons': TRAIN, 'conditional_known_absences': True}, folder/f'model_{stat}_{label}.joblib')
                test_predictions[label] = prediction
                for month, indexes in test.groupby(test.date.dt.to_period('M')).indices.items():
                    monthly.append({'stat': stat, 'model': label, 'month': str(month), 'rows': len(indexes),
                                    'mae': float(errors[indexes].mean())})
                results.append(row)
                print(f'TEST {stat} {label} ({candidate_id}): MAE={row["mae"]:.4f}, accuracy={row["direction_confidence"]["accuracy"]:.2%}', flush=True)
            primary_label = next(label for label in test_predictions if selected[stat][label] == selected[stat]['primary_development_choice'])
            primary_errors = abs(test[stat].to_numpy()-test_predictions[primary_label])
            for row in [r for r in results if r['stat'] == stat]:
                errors = abs(test[stat].to_numpy()-test_predictions[row['model']])
                row['mae_change_vs_development_choice'] = float(np.mean(errors-primary_errors))
                row['paired_95pct_vs_development_choice'] = paired_date_interval(test.date, primary_errors, errors)
    import catboost, sklearn, xgboost
    report = {'database_writes': False, 'production_models_overwritten': False,
              'source': provenance, 'frozen_comparison': frozen, 'coverage': coverage,
              'versions': {'catboost': catboost.__version__, 'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__},
              'limitations': ['2025-26 has been evaluated before; subsequent choices require confirmation on future data.',
                              'Hyperparameters, loss, family and blend use only 2023-24/2024-25 expanding validation; all final regressors train the first three seasons.',
                              'Tuning and empirical confidence calibration reuse development predictions, so calibration is not independently validated until test assessment.',
                              'Fixed same feature lists/cohorts for all models. Known historical absences and current position metadata remain retrospective inputs.',
                              'Accuracy uses season-to-date average proxies conditional on non-pushes, not sportsbook lines or profitability.',
                              'Interval calibration transfers from smaller earlier models to final refits; coverage is empirical rather than guaranteed.',
                              'MAE and squared-error objectives target different point summaries; compare RMSE, bias and probability calibration alongside MAE.',
                              'Date-block intervals are exploratory without multiple-testing or tuning-uncertainty adjustment.'],
              'results': results}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'candidate_id': r['candidate_id'],
                   'development_choice': r['primary_development_choice'],
                   'mae': r['mae'], 'rmse': r['rmse'], 'bias': r['bias'], 'r2': r['r2'],
                   'accuracy': r['direction_confidence']['accuracy'], 'brier': r['direction_confidence']['brier_score'],
                   'log_loss': r['direction_confidence']['log_loss'], 'coverage_90': r['intervals'][-1]['observed_coverage'],
                   'width_90': r['intervals'][-1]['mean_width'], 'mae_change_vs_linear': r['mae_change_vs_linear'],
                   'paired_95pct_vs_linear': r['paired_95pct_vs_linear']} for r in results]).to_csv(folder/'summary.csv', index=False)
    pd.DataFrame([{'stat':r['stat'],'model':r['model'],**b} for r in results for b in r['direction_confidence']['buckets']]).to_csv(folder/'confidence_buckets.csv',index=False)
    pd.DataFrame([{'stat':r['stat'],'model':r['model'],**t} for r in results for t in r['direction_confidence']['thresholds']]).to_csv(folder/'confidence_thresholds.csv',index=False)
    pd.DataFrame(monthly).to_csv(folder/'monthly_results.csv',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)


if __name__ == '__main__':
    main()
