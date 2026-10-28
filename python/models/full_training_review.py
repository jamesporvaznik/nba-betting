"""Freeze features on 2022-25, refit all three seasons, evaluate 2025-26 once.

Absence inputs are historical known-absence scenarios. Confidence uses expanding
out-of-time development predictions and transfers to refitted models; this is
empirical temporal calibration, not a strict split-conformal coverage guarantee.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.evaluation.metrics import metrics, paired_date_interval
from models.features.absences import absence_groups
from models.features.engineering import eligible_rows, get_feature_cols
from models.features.matchup import experimental_groups
from models.features.offense import offense_groups
from models.features.rebounding import rebound_columns

TRAIN = ['2022-23', '2023-24', '2024-25']
TEST = '2025-26'
STATS = ['points', 'rebounds', 'assists']
KEYS = ['playerid', 'season', 'date', 'team', 'opponent']


from models.evaluation.confidence import calibration_fit, assess
from models.data.snapshot import capture_training_snapshot
from models.features.selected import build_selected_features

def capture():
    return capture_training_snapshot(TRAIN + [TEST])


def estimator(name):
    return LinearRegression() if name == 'linear' else XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, random_state=42, n_jobs=1)


def packages(stat):
    base = get_feature_cols(stat) + offense_groups(stat)['shot_volume']
    core = base + absence_groups()['position_and_adaptation']
    role = experimental_groups(stat)['opportunity']
    result = {'baseline': base, 'absence_core': core, 'absence_role': core + role}
    if stat == 'rebounds':
        team = rebound_columns()['team_rebounding']
        result.update({'absence_team': core + team, 'absence_role_team': core + role + team})
    return result


def development_folds(frame):
    """Only the first three seasons can participate in feature selection/calibration."""
    for season in TRAIN[1:]:
        train = frame[frame.season.isin([s for s in TRAIN if s < season])]
        validation = frame[frame.season == season]
        if train.empty or validation.empty or train.date.max() >= validation.date.min():
            raise ValueError('Invalid development chronology')
        yield season, train, validation


def select_package(rows):
    """Prefer core unless an addition improves both folds and mean relative MAE >=.1%."""
    scores = pd.DataFrame(rows)
    reference = scores[scores.package.eq('absence_core')].set_index('season').mae
    candidates = [('absence_core', 0., 0)]
    for name, group in scores.groupby('package', sort=False):
        if name in ['baseline', 'absence_core']:
            continue
        relative = group.set_index('season').mae / reference - 1
        if (relative < 0).all() and relative.mean() <= -.001:
            candidates.append((name, float(relative.mean()), int(group.feature_count.iloc[0])))
    return min(candidates, key=lambda r: (r[1], r[2]))[0]






def prepare_features(snapshot, expected_seasons=None):
    required = TRAIN + [TEST] if expected_seasons is None else expected_seasons
    if {r['season'] for r in snapshot['players']} != set(required):
        raise ValueError('Unexpected modeling seasons')
    print('Engineering histories and known-absence features.', flush=True)
    return eligible_rows(build_selected_features(snapshot))


def run(snapshot, folder):
    eligible = prepare_features(snapshot)
    selections, development, coverage, all_results, naive, monthly = {}, [], [], [], [], []
    frames, stored = {}, {}
    with threadpool_limits(limits=1):
        # Freeze all feature choices using only pre-test seasons before any test prediction.
        for stat in STATS:
            candidates = packages(stat)
            columns = list(dict.fromkeys(c for cols in candidates.values() for c in cols))
            common = eligible.dropna(subset=columns + [stat, f'{stat}_scale'])
            if not np.isfinite(common[columns].to_numpy(float)).all():
                raise ValueError('Nonfinite model inputs')
            frames[stat] = common
            stat_rows = []
            for season, train, validation in development_folds(common):
                for package, inputs in candidates.items():
                    model = estimator('linear').fit(train[inputs], train[stat])
                    prediction = model.predict(validation[inputs])
                    row = {'stat': stat, 'season': season, 'package': package, 'feature_count': len(inputs),
                           'mae': float(np.mean(abs(validation[stat].to_numpy() - prediction)))}
                    stat_rows.append(row)
                    stored[(stat, package, season)] = prediction
            selected = select_package(stat_rows)
            selections[stat] = {'package': selected, 'features': candidates[selected], 'primary_model': 'linear'}
            development.extend(stat_rows)
            print(f'Frozen {stat}: {selected}, {len(candidates[selected])} features.', flush=True)
        manifest = {'selected_before_test_predictions': True, 'training_seasons': TRAIN, 'test_season': TEST,
                    'selection_rule': 'Primary OLS fixed before test; core absences retained unless an addition lowers OLS MAE in both expanding folds and average relative MAE by >=0.1%. Pick largest qualifying average reduction.',
                    'selections': selections, 'development': development}
        (folder/'frozen_features.json').write_text(json.dumps(manifest, indent=2))
        for stat in STATS:
            common = frames[stat]
            training, test = common[common.season.isin(TRAIN)], common[common.season.eq(TEST)]
            if training.empty or test.empty or training.date.max() >= test.date.min():
                raise ValueError('Invalid final chronology')
            coverage.append({'stat': stat, 'train_rows': len(training), 'test_rows': len(test),
                             'train_players': int(training.playerid.nunique()), 'test_players': int(test.playerid.nunique()),
                             'train_dates': [str(training.date.min().date()), str(training.date.max().date())],
                             'test_dates': [str(test.date.min().date()), str(test.date.max().date())],
                             'test_eligible_before_feature_filter': int(eligible.season.eq(TEST).sum()),
                             'rows_with_absences': int(test.absent_count.gt(0).sum())})
            actual, lines = test[stat].to_numpy(), test[f'{stat}_avg_season'].to_numpy()
            for label, predictions in [('last5', test[f'{stat}_avg5'].to_numpy()), ('season_average', lines)]:
                naive.append({'stat': stat, 'model': label, **metrics(actual, predictions, lines)})
            errors_by_model = {}
            for variant, package in [('baseline', 'baseline'), ('optimized', selections[stat]['package'])]:
                inputs = packages(stat)[package]
                for name in ['linear', 'xgboost']:
                    calibration_rows = []
                    for season, train, validation in development_folds(common):
                        if name == 'linear':
                            prediction = stored[(stat, package, season)]
                        else:
                            prediction = estimator(name).fit(train[inputs], train[stat]).predict(validation[inputs])
                        part = validation[KEYS].copy()
                        part['actual'], part['prediction'] = validation[stat].to_numpy(), prediction
                        part['line'], part['scale'] = validation[f'{stat}_avg_season'].to_numpy(), validation[f'{stat}_scale'].to_numpy()
                        part['training_last_date'] = str(train.date.max().date())
                        calibration_rows.append(part)
                    calibration = pd.concat(calibration_rows, ignore_index=True)
                    classifier, quantiles, cal_rate = calibration_fit(calibration)
                    model = estimator(name).fit(training[inputs], training[stat])
                    prediction = model.predict(test[inputs])
                    result, output = assess(test, prediction, stat, classifier, quantiles, cal_rate)
                    result.update({'stat': stat, 'model': name, 'variant': variant, 'package': package,
                                   'features': inputs, 'train_rows': len(training), 'calibration_rows': len(calibration), 'test_rows': len(test)})
                    errors = abs(actual - prediction)
                    errors_by_model[(name, variant)] = errors
                    if variant == 'optimized':
                        baseline = errors_by_model[(name, 'baseline')]
                        result.update({'mae_change_vs_baseline': float(np.mean(errors - baseline)),
                                       'paired_95pct_vs_baseline': paired_date_interval(test.date, baseline, errors),
                                       'mae_change_vs_last5': float(np.mean(errors - abs(actual - test[f'{stat}_avg5'].to_numpy())))})
                    for month, indexes in test.groupby(test.date.dt.to_period('M')).indices.items():
                        monthly.append({'stat': stat, 'model': name, 'variant': variant, 'month': str(month),
                                        'rows': len(indexes), **metrics(actual[indexes], prediction[indexes], lines[indexes])})
                    output.to_csv(folder/f'predictions_{stat}_{name}_{variant}.csv', index=False)
                    calibration.to_csv(folder/f'calibration_{stat}_{name}_{variant}.csv', index=False)
                    joblib.dump({'regressor': model, 'direction_calibrator': classifier, 'interval_quantiles': quantiles,
                                 'features': inputs, 'training_seasons': TRAIN, 'variant': variant,
                                 'conditional_known_absences': variant == 'optimized',
                                 'scale': 'Prior ten-appearance stat standard deviation, floor one'},
                                folder/f'model_{stat}_{name}_{variant}.joblib')
                    all_results.append(result)
                    print(f'{stat} {name} {variant}: MAE={result["mae"]:.4f}, calibrated proxy accuracy={result["direction_confidence"]["accuracy"]:.2%}', flush=True)
    report = {'captured_at': snapshot.get('captured_at'), 'database_writes': False, 'production_models_overwritten': False,
              'training_seasons': TRAIN, 'test_season': TEST, 'features_frozen_before_test_predictions': True,
              'primary_model': 'linear', 'coverage': coverage, 'selections': selections,
              'confidence_method': 'Expanding out-of-time predictions on 2023-24 and 2024-25 train logistic direction calibrators and scaled absolute-residual quantiles. Final regressors refit all three seasons; calibration transfers without using test labels.',
              'limitations': ['2025-26 was evaluated earlier in the baseline review; this is not a pristine untouched holdout.',
                              'Optimized results assume actual game absences are supplied. Absence rows and observed stint boundaries are retrospective, not timestamped pregame reports.',
                              'Positions are current master data rather than historical snapshots; stint boundary/transition gaps remain excluded.',
                              'Earlier completed test games update rolling histories; all regressors and calibrators stay fixed throughout evaluation.',
                              'Eligibility requires 20 prior season appearances and >=10 prior average minutes; predicts observed appearances, not DNP outcomes.',
                              'Confidence concerns exceeding a season-to-date average proxy conditional on a non-push, not actual betting lines.',
                              'OOF calibration transfers to refitted models. Nominal interval coverage is empirical, not a strict split-conformal guarantee under refitting or time dependence.',
                              'Development feature selection and multiple comparisons make date-block bootstrap intervals exploratory.'],
              'naive_baselines': naive, 'results': all_results}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'variant': r['variant'], 'package': r['package'],
                   'mae': r['mae'], 'rmse': r['rmse'], 'bias': r['bias'], 'r2': r['r2'],
                   'raw_proxy_accuracy': r['proxy_accuracy'], 'calibrated_proxy_accuracy': r['direction_confidence']['accuracy'],
                   'brier': r['direction_confidence']['brier_score'], 'log_loss': r['direction_confidence']['log_loss'],
                   'constant_direction_accuracy': r['direction_confidence']['constant_direction_accuracy'],
                   'coverage_90': r['intervals'][-1]['observed_coverage'], 'width_90': r['intervals'][-1]['mean_width'],
                   'train_rows': r['train_rows'], 'test_rows': r['test_rows']} for r in all_results]).to_csv(folder/'summary.csv', index=False)
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'variant': r['variant'], **bucket}
                  for r in all_results for bucket in r['direction_confidence']['buckets']]).to_csv(folder/'confidence_buckets.csv', index=False)
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'variant': r['variant'], **threshold}
                  for r in all_results for threshold in r['direction_confidence']['thresholds']]).to_csv(folder/'confidence_thresholds.csv', index=False)
    pd.DataFrame(monthly).to_csv(folder/'monthly_results.csv', index=False)
    pd.DataFrame(development).to_csv(folder/'development_selection.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else capture()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir/'source_snapshot.json'
    source.write_text(json.dumps(snapshot))
    run(snapshot, args.output_dir)
    report = args.output_dir/'report.json'
    data = json.loads(report.read_text())
    data.update({'source': 'live_database' if not args.snapshot else str(args.snapshot),
                 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    report.write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
