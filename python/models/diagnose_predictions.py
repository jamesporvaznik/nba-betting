"""Diagnose saved development predictions without refitting or accessing Supabase.

Regression feature comparisons use both expanding validation folds. Confidence
assessment uses its own fixed-training/calibration model family for 2024-25;
these are never combined with predictions from a differently trained model.
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

import numpy as np
import pandas as pd

from models.features.engineering import engineer_features
from models.features.offense import add_offense_features

KEYS = ['playerid', 'season', 'date', 'team', 'opponent']
SEGMENTS = ['month', 'scoring_level', 'recent_minutes', 'minutes_role', 'shot_role']


def add_segments(frame):
    """Prespecified slices use prior-only profiles, never current outcomes."""
    result = frame.copy()
    result['month'] = pd.to_datetime(result.date).dt.strftime('%Y-%m')
    result['scoring_level'] = pd.cut(result.points_avg_season, [-np.inf, 10, 20, np.inf],
                                     right=False, labels=['under 10 PPG', '10–20 PPG', '20+ PPG']).astype(str)
    result['recent_minutes'] = pd.cut(result.minutes_avg5, [-np.inf, 20, 30, np.inf],
                                      right=False, labels=['under 20 min', '20–30 min', '30+ min']).astype(str)
    trend = result.minutes_avg5-result.minutes_avg_season
    result['minutes_role'] = np.select([trend <= -5, trend >= 5], ['falling (5+ min)', 'rising (5+ min)'], default='stable (<5 min change)')
    shot_trend = (result.fga_prior5-result.fga_priorseason) / result.fga_priorseason.clip(lower=1)
    result['shot_role'] = np.select([shot_trend <= -.25, shot_trend >= .25],
                                    ['falling (25%+)', 'rising (25%+)'], default='stable (<25% change)')
    return result


def attach(predictions, features, stat):
    predictions = predictions.copy()
    predictions['date'] = pd.to_datetime(predictions.date)
    columns = list(dict.fromkeys(KEYS + SEGMENTS + [stat, 'points_avg_season', 'minutes_avg5',
                          f'{stat}_avg5', f'{stat}_avg_season']))
    combined = predictions.merge(features[columns], on=KEYS, how='left',
                                  validate='one_to_one', suffixes=('', '_source'), indicator=True)
    if not combined['_merge'].eq('both').all() or not np.array_equal(combined[stat], combined[f'{stat}_source']):
        raise ValueError('Saved predictions differ from source appearance coverage or outcomes')
    return combined.drop(columns=['_merge', f'{stat}_source'])


def bootstrap_mean(dates, values, repeats=1000):
    """Resample whole dates, retaining all player appearances together."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    daily = pd.DataFrame({'date': np.asarray(dates), 'value': values}).groupby('date').value.agg(['sum', 'count'])
    if len(daily) < 2:
        return [None, None]
    sample = np.random.default_rng(2026).integers(0, len(daily), size=(repeats, len(daily)))
    means = daily['sum'].to_numpy()[sample].sum(axis=1) / daily['count'].to_numpy()[sample].sum(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def base_summary(frame):
    return {'appearances': len(frame), 'players': int(frame.playerid.nunique()), 'dates': int(frame.date.nunique()),
            'small_sample': bool(len(frame) < 200 or frame.playerid.nunique() < 20 or frame.date.nunique() < 10)}


def regression_summary(frame, stat):
    y = frame[stat].to_numpy()
    base = np.abs(y-frame.linear_baseline.to_numpy())
    new = np.abs(y-frame.linear_shot_volume.to_numpy())
    result = {**base_summary(frame), 'linear_baseline_mae': float(base.mean()),
              'linear_shots_mae': float(new.mean()),
              'last5_mae': float(np.abs(y-frame[f'{stat}_avg5'].to_numpy()).mean()),
              'season_average_mae': float(np.abs(y-frame[f'{stat}_avg_season'].to_numpy()).mean()),
              'shots_minus_baseline_mae': float((new-base).mean()),
              'mae_change_95pct_date_bootstrap': bootstrap_mean(frame.date, new-base)}
    return result


def confidence_summary(frame, stat, constant_over):
    result = base_summary(frame)
    y = frame[stat].to_numpy()
    lines = frame.pregame_average_proxy_line.to_numpy()
    valid = ~np.isclose(y, lines, rtol=0, atol=1e-10)
    p = frame.probability_over_proxy_nonpush.to_numpy()[valid]
    outcome = y[valid] > lines[valid]
    correct = (p > .5) == outcome
    constant_correct = constant_over == outcome
    dates = frame.loc[valid, 'date']
    result.update({'nonpush_predictions': int(valid.sum()), 'actual_pushes': int((~valid).sum()),
                   'mean_direction_confidence': float(np.maximum(p, 1-p).mean()) if len(p) else None,
                   'direction_accuracy': float(correct.mean()) if len(p) else None,
                   'accuracy_95pct_date_bootstrap': bootstrap_mean(dates, correct),
                   'constant_direction_accuracy': float(constant_correct.mean()) if len(p) else None,
                   'accuracy_gain_vs_constant': float(np.mean(correct.astype(float)-constant_correct)) if len(p) else None,
                   'accuracy_gain_95pct_date_bootstrap': bootstrap_mean(dates, correct.astype(float)-constant_correct),
                   'brier': float(np.mean((p-outcome)**2)) if len(p) else None,
                   'coverage_90': float(((y >= frame.lower_90) & (y <= frame.upper_90)).mean()),
                   'mean_width_90': float((frame.upper_90-frame.lower_90).mean())})
    baseline_correct = (frame.baseline_probability_over_proxy.to_numpy()[valid] > .5) == outcome
    result['calibrated_baseline_accuracy'] = float(baseline_correct.mean()) if len(p) else None
    result['accuracy_gain_vs_calibrated_baseline'] = float(np.mean(correct.astype(float)-baseline_correct)) if len(p) else None
    result['gain_vs_calibrated_baseline_95pct'] = bootstrap_mean(dates, correct.astype(float)-baseline_correct)
    recent = frame[f'{stat}_avg5'].to_numpy()
    paired = valid & ~np.isclose(recent, lines, rtol=0, atol=1e-10)
    recent_correct = (recent[paired] > lines[paired]) == (y[paired] > lines[paired])
    paired_model_correct = (frame.probability_over_proxy_nonpush.to_numpy()[paired] > .5) == (y[paired] > lines[paired])
    result.update({'last5_comparison_predictions': int(paired.sum()),
                   'last5_direction_accuracy': float(recent_correct.mean()) if paired.any() else None,
                   'model_accuracy_on_last5_comparison': float(paired_model_correct.mean()) if paired.any() else None,
                   'accuracy_gain_vs_last5': float(np.mean(paired_model_correct.astype(float)-recent_correct)) if paired.any() else None,
                   'gain_vs_last5_95pct': bootstrap_mean(frame.loc[paired, 'date'], paired_model_correct.astype(float)-recent_correct)})
    return result


def slices(frame, include_confidence=False):
    yield 'overall', 'all', frame
    for segment in SEGMENTS:
        for label, group in frame.groupby(segment, observed=True, sort=True):
            yield segment, str(label), group
    if include_confidence:
        confidence = frame.chosen_direction_confidence_nonpush
        bins = pd.cut(confidence, [.5, .55, .6, .65, .7, .8, .9, 1.000001], right=False,
                      labels=['50–55%', '55–60%', '60–65%', '65–70%', '70–80%', '80–90%', '90–100%'])
        for label, group in frame.groupby(bins, observed=True):
            yield 'confidence', str(label), group
        # Cross-check calibration across time and roles, not only pooled bins.
        for segment in ['month', 'minutes_role']:
            for (label, bucket), group in frame.groupby([frame[segment], bins], observed=True, sort=True):
                yield f'{segment}_by_confidence', f'{label} | {bucket}', group


def run(experiment, confidence, folder):
    snapshot = json.loads((experiment/'source_snapshot.json').read_text())
    raw = pd.DataFrame(snapshot['players'])
    if set(raw.season) - {'2022-23', '2023-24', '2024-25'}:
        raise ValueError('Do not include 2025-26 or later data')
    with redirect_stdout(io.StringIO()):
        features = add_segments(add_offense_features(engineer_features(raw)))
    confidence_report = json.loads((confidence/'report.json').read_text())
    source_hash = hashlib.sha256((experiment/'source_snapshot.json').read_bytes()).hexdigest()
    if confidence_report.get('source_snapshot_sha256') != source_hash:
        raise ValueError('Confidence and feature comparisons must reference the same source snapshot')
    regression_rows, confidence_rows = [], []
    for season in ['2023-24', '2024-25']:
        for stat in ['points', 'rebounds', 'assists']:
            predictions = pd.read_csv(experiment/f'predictions_{season}_{stat}.csv')
            if not predictions.season.eq(season).all():
                raise ValueError('Unexpected regression prediction season')
            frame = attach(predictions, features, stat)
            for segment, label, group in slices(frame):
                regression_rows.append({'season': season, 'stat': stat, 'segment': segment,
                                        'group': label, **regression_summary(group, stat)})
    for stat in ['points', 'rebounds', 'assists']:
        predictions = pd.read_csv(confidence/f'predictions_{stat}_linear_shot_volume.csv')
        baseline_predictions = pd.read_csv(confidence/f'predictions_{stat}_linear_baseline.csv')
        predictions = predictions.merge(baseline_predictions[KEYS + ['probability_over_proxy_nonpush']].rename(
            columns={'probability_over_proxy_nonpush': 'baseline_probability_over_proxy'}),
            on=KEYS, validate='one_to_one', how='left')
        if predictions.baseline_probability_over_proxy.isna().any():
            raise ValueError('Calibrated baseline and shot-volume prediction cohorts differ')
        if not predictions.season.eq('2024-25').all():
            raise ValueError('Unexpected confidence prediction season')
        frame = attach(predictions, features, stat)
        baseline = next(r for r in confidence_report['results'] if r['stat'] == stat and r['model'] == 'linear' and r['group'] == 'shot_volume')
        constant_over = baseline['direction_confidence']['constant_calibration_base_rate'] > .5
        for segment, label, group in slices(frame, include_confidence=True):
            confidence_rows.append({'season': '2024-25', 'stat': stat, 'segment': segment,
                                    'group': label, **confidence_summary(group, stat, constant_over)})
    report = {'database_writes': False, 'model_refits': 0, 'held_out_2025_26_read_or_scored': False,
              'sources': {'regression_experiment': str(experiment), 'confidence_assessment': str(confidence)},
              'method': 'Paired/date-block bootstrap with 1000 resamples. Prespecified prior-only slices; no current outcomes used for scoring level, minutes, or role groups. Small-sample flag if <200 appearances, <20 players, or <10 dates. Exploratory intervals are not corrected for multiple slices or prior model selection.',
              'model_families': {'regression': 'Original expanding-season validation models: train preceding seasons for each validation season.',
                                 'confidence': 'Linear shot-volume regressor trained only 2022-23, uncertainty calibrated 2023-24, assessed 2024-25. Constant direction learned from calibration season; no slice-specific hindsight baseline.'},
              'limitations': ['Only eligible players with 20 prior season appearances and 10 prior average minutes; earliest season games are absent.',
                              '2024-25 has already influenced development; not a fresh final holdout.',
                              'Repeated players can remain dependent across dates; date-block intervals do not capture every dependence source.',
                              'Direction accuracy uses an average-based proxy conditional on a non-push, not betting lines.',
                              'Role-change groups are historical indicators, not injury-aware role forecasts.'],
              'regression': regression_rows, 'confidence': confidence_rows}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(regression_rows).to_csv(folder/'regression_slices.csv', index=False)
    pd.DataFrame(confidence_rows).to_csv(folder/'confidence_slices.csv', index=False)
    print(f'Saved {len(regression_rows)} regression slices and {len(confidence_rows)} confidence slices. No refits, database access, or 2025-26 data.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True, type=Path)
    parser.add_argument('--confidence-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.experiment_dir, args.confidence_dir, args.output_dir)


if __name__ == '__main__':
    main()
