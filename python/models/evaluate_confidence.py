"""Chronological calibration of intervals and average-proxy direction confidence.

Train fixed regressors on 2022-23, calibrate on 2023-24, assess on 2024-25.
Use a saved offense snapshot only; never read the database or 2025-26 inputs.
2024-25 has already been explored, so this remains a development assessment.
"""

import argparse
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.experiment_features import metrics, SEASONS
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.matchup import add_matchup_features
from models.features.offense import add_offense_features, offense_groups








from models.evaluation.confidence import conformity_quantile, margin, confidence_buckets

def run(snapshot, folder):
    players = pd.DataFrame(snapshot['players'])
    teams = pd.DataFrame(snapshot['teams'])
    if set(players.season) - set(SEASONS) or set(teams.season) - set(SEASONS):
        raise ValueError('Only pre-2025-26 development seasons are allowed')
    with redirect_stdout(io.StringIO()):
        features = add_offense_features(add_matchup_features(engineer_features(players), teams))
    for stat in ['points', 'rebounds', 'assists']:
        # A fixed, observable scale for heteroscedastic scoring variability.
        features[f'{stat}_uncertainty_scale'] = features.groupby(['playerid', 'season'])[stat].transform(
            lambda x: x.shift(1).rolling(10, min_periods=2).std()).clip(lower=1)
    eligible = eligible_rows(features)
    results = []
    with threadpool_limits(limits=1):
        for stat in ['points', 'rebounds', 'assists']:
            base = get_feature_cols(stat)
            shots = offense_groups(stat)['shot_volume']
            scale_col = f'{stat}_uncertainty_scale'
            common = eligible.dropna(subset=base + shots + [stat, scale_col])
            train, calibration, evaluation = [common[common.season == season] for season in SEASONS]
            if (any(x.empty for x in (train, calibration, evaluation))
                    or train.date.max() >= calibration.date.min()
                    or calibration.date.max() >= evaluation.date.min()):
                raise ValueError('Invalid chronological calibration split')
            for group, extras in [('baseline', []), ('shot_volume', shots)]:
                columns = base + extras
                for name in ['linear', 'xgboost']:
                    model = LinearRegression() if name == 'linear' else XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=.05,
                        subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=1)
                    model.fit(train[columns], train[stat])
                    cal_pred = model.predict(calibration[columns])
                    pred = model.predict(evaluation[columns])
                    cal_y, actual = calibration[stat].to_numpy(), evaluation[stat].to_numpy()
                    cal_scale, scale = calibration[scale_col].to_numpy(), evaluation[scale_col].to_numpy()
                    cal_lines = calibration[f'{stat}_avg_season'].to_numpy()
                    lines = evaluation[f'{stat}_avg_season'].to_numpy()
                    scores = np.abs(cal_y-cal_pred) / cal_scale
                    result = {'stat': stat, 'model': name, 'group': group, 'features': columns,
                              'train_rows': len(train), 'calibration_rows': len(calibration), 'evaluation_rows': len(evaluation),
                              **metrics(actual, pred, lines), 'intervals': []}
                    output = evaluation[['playerid', 'season', 'date', 'team', 'opponent', stat]].copy()
                    output['prediction'] = pred
                    output['pregame_average_proxy_line'] = lines
                    output['uncertainty_scale'] = scale
                    for level in [.50, .80, .90]:
                        q = conformity_quantile(scores, level)
                        lower = np.maximum(0, pred-q*scale)
                        upper = np.maximum(0, pred+q*scale)
                        covered = (actual >= lower) & (actual <= upper)
                        result['intervals'].append({'nominal_coverage': level, 'observed_coverage': float(covered.mean()),
                                                    'mean_width': float((upper-lower).mean()),
                                                    'median_width': float(np.median(upper-lower)), 'score_quantile': q})
                        output[f'lower_{int(level*100)}'] = lower
                        output[f'upper_{int(level*100)}'] = upper
                    cal_valid = ~np.isclose(cal_y, cal_lines, rtol=0, atol=1e-10)
                    classifier = LogisticRegression(C=1e6, max_iter=1000)
                    classifier.fit(margin(cal_pred, cal_lines, cal_scale)[cal_valid],
                                   (cal_y[cal_valid] > cal_lines[cal_valid]).astype(int))
                    probabilities = classifier.predict_proba(margin(pred, lines, scale))[:, 1]
                    valid = ~np.isclose(actual, lines, rtol=0, atol=1e-10)
                    y = (actual[valid] > lines[valid]).astype(int)
                    cal_base = float((cal_y[cal_valid] > cal_lines[cal_valid]).mean())
                    result['direction_confidence'] = {
                        'interpretation': 'Fitted probability of exceeding the pregame season-average proxy conditional on a non-push; not sportsbook probability.',
                        'nonpush_rows': int(valid.sum()),
                        'accuracy': float(np.mean((probabilities[valid] > .5) == y)),
                        'brier_score': float(brier_score_loss(y, probabilities[valid])),
                        'constant_calibration_base_rate': cal_base,
                        'constant_calibration_base_rate_accuracy': float(np.mean((cal_base > .5) == y)),
                        'constant_calibration_base_rate_brier': float(brier_score_loss(y, np.full(len(y), cal_base))),
                        'log_loss': float(log_loss(y, probabilities[valid], labels=[0, 1])),
                        'calibration_intercept': float(classifier.intercept_[0]),
                        'calibration_slope': float(classifier.coef_[0, 0]),
                        'buckets': confidence_buckets(actual, lines, probabilities)}
                    output['probability_over_proxy_nonpush'] = probabilities
                    output['chosen_direction_confidence_nonpush'] = np.maximum(probabilities, 1-probabilities)
                    output.to_csv(folder / f'predictions_{stat}_{name}_{group}.csv', index=False)
                    # New calibration artifacts only; previous model files stay intact.
                    joblib.dump({'regressor': model, 'direction_calibrator': classifier,
                                 'interval_quantiles': {str(i['nominal_coverage']): i['score_quantile'] for i in result['intervals']},
                                 'features': columns, 'scale': 'Prior 10-appearance stat std, floor 1'},
                                folder / f'calibrated_{stat}_{name}_{group}.joblib')
                    results.append(result)
                    print(f'{stat} {name} {group}: MAE={result["mae"]:.3f}; proxy direction={result["proxy_accuracy"]:.1%}; calibrated direction={result["direction_confidence"]["accuracy"]:.1%}; 90% interval coverage={result["intervals"][-1]["observed_coverage"]:.1%}', flush=True)
    report = {'database_writes': False, 'train_season': '2022-23', 'calibration_season': '2023-24', 'evaluation_season': '2024-25',
              'held_out_2025_26_read_or_scored': False,
              'limitations': ['2024-25 was already used for development feature comparisons; this is not a fresh final holdout.',
                              'Time dependence and season drift mean nominal conformal coverage is not guaranteed; observed coverage is reported.',
                              'No refitting on calibration rows after calibrating. Fewer training rows than the prior two-season training run, so results are not directly comparable.',
                              'Probabilities concern an average-based proxy conditional on a non-push, not real betting lines.'],
              'results': results}
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'group': r['group'], 'mae': r['mae'],
                   'proxy_direction_accuracy': r['proxy_accuracy'],
                   'calibrated_direction_accuracy': r['direction_confidence']['accuracy'],
                   'constant_direction_accuracy': r['direction_confidence']['constant_calibration_base_rate_accuracy'],
                   'coverage_90': r['intervals'][-1]['observed_coverage'],
                   'mean_interval_width_90': r['intervals'][-1]['mean_width']} for r in results]).to_csv(folder / 'summary.csv', index=False)
    pd.DataFrame([{'stat': r['stat'], 'model': r['model'], 'group': r['group'], **b}
                  for r in results for b in r['direction_confidence']['buckets']]).to_csv(folder / 'confidence_buckets.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(snapshot, args.output_dir)
    report_path = args.output_dir / 'report.json'
    report = json.loads(report_path.read_text())
    report['source_snapshot'] = str(args.snapshot)
    report['source_snapshot_sha256'] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
