"""Test lagged playtype profiles and matchup interactions without season leakage.

Existing profiles start in 2022-23. First usable target season is 2023-24:
train its eligible games before 2024-02-01 and score later games. Then train all
eligible 2023-24 and score 2024-25. Every variant shares rows and train-only
imputation; no 2025-26 input is read. Baselines are retrained on these new splits.
--profile-timing completed_current is a separate, explicitly leaky retrospective
diagnostic. Its scores must not be presented as attainable pregame performance.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.experiment_features import metrics, paired_date_interval, SEASONS
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.offense import add_offense_features, offense_groups
from models.features.playtype_matchup import add_playtype_matchups, playtype_columns
from nba.repair_shot_locations import connection


def capture_profiles(profile_timing='previous'):
    queries = {
        'players': 'SELECT player_id,season,play_type,poss_pct FROM public.player_playtype_stats WHERE season = ANY(%s) ORDER BY season,player_id,play_type',
        'defense': 'SELECT team_abbreviation AS opponent,season,play_type,points_allowed_per_possession AS ppp,possessions_defended FROM public.team_defensive_playtype_stats WHERE season = ANY(%s) ORDER BY season,team_abbreviation,play_type',
    }
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            for name, query in queries.items():
                seasons = ['2022-23', '2023-24'] if profile_timing == 'previous' else ['2023-24', '2024-25']
                c.execute(query, (seasons,))
                snapshot[name] = [dict(r) for r in c.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def chronological_folds(frame):
    cutoff = pd.Timestamp('2024-02-01')
    first = frame[frame.season == '2023-24']
    return [('late_2023-24', first[first.date < cutoff], first[first.date >= cutoff]),
            ('full_2024-25', first, frame[frame.season == '2024-25'])]


def run(game_snapshot, profiles, folder, profile_timing='previous'):
    raw = pd.DataFrame(game_snapshot['players'])
    if set(raw.season)-set(SEASONS):
        raise ValueError('No 2025-26 or other game seasons allowed')
    allowed_profiles = {'2022-23', '2023-24'} if profile_timing == 'previous' else {'2023-24', '2024-25'}
    for records in [profiles['players'], profiles['defense']]:
        if {r['season'] for r in records} - allowed_profiles:
            raise ValueError('Unexpected profile seasons for this experiment')
    with redirect_stdout(io.StringIO()):
        features = add_offense_features(engineer_features(raw))
        features = add_playtype_matchups(features, pd.DataFrame(profiles['players']), pd.DataFrame(profiles['defense']), profile_timing)
    base = get_feature_cols('points') + offense_groups('points')['shot_volume']
    eligible = eligible_rows(features).dropna(subset=base + ['points'])
    if not np.isfinite(eligible[base + ['points']].to_numpy(dtype=float)).all():
        raise ValueError('Invalid baseline input')
    groups, results, coverage = playtype_columns(profile_timing), [], []
    prefix = 'prior' if profile_timing == 'previous' else 'retrospective'
    with threadpool_limits(limits=1):
        for label, train, validation in chronological_folds(eligible):
            if train.empty or validation.empty or train.date.max() >= validation.date.min():
                raise ValueError('Invalid chronological profile split')
            coverage.append({'fold': label, 'train_rows': len(train), 'validation_rows': len(validation),
                             'train_first_date': str(train.date.min().date()), 'train_last_date': str(train.date.max().date()),
                             'validation_first_date': str(validation.date.min().date()), 'validation_last_date': str(validation.date.max().date()),
                             'train_player_profile_missing': int(train[f'{prefix}_player_profile_available'].eq(0).sum()),
                             'validation_player_profile_missing': int(validation[f'{prefix}_player_profile_available'].eq(0).sum()),
                             'validation_opponent_profile_missing': int(validation[f'{prefix}_opponent_profile_available'].eq(0).sum())})
            actual = validation.points.to_numpy()
            lines = validation.points_avg_season.to_numpy()
            predictions = validation[['playerid', 'season', 'date', 'team', 'opponent', 'points',
                                       f'{prefix}_player_profile_available', f'{prefix}_opponent_profile_available']].copy()
            references = {}
            for group, extras in groups.items():
                columns = base + extras
                for name in ['linear', 'xgboost']:
                    model = LinearRegression() if name == 'linear' else XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
                        colsample_bytree=.8, random_state=42, n_jobs=1)
                    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)
                    pipeline.fit(train[columns], train.points)
                    pred = pipeline.predict(validation[columns])
                    errors = np.abs(actual-pred)
                    predictions[f'{name}_{group}'] = pred
                    result = {'fold': label, 'model': name, 'variant': group, 'features': columns,
                              'train_rows': len(train), 'validation_rows': len(validation), **metrics(actual, pred, lines)}
                    if group == 'baseline':
                        references[name] = errors
                    result['mae_change_vs_baseline'] = float(errors.mean()-references[name].mean())
                    result['paired_95pct_vs_baseline'] = paired_date_interval(validation.date, references[name], errors)
                    if group == 'both_profiles':
                        references[f'{name}_both'] = errors
                    if group == 'interactions':
                        result['mae_change_vs_both_profiles'] = float(errors.mean()-references[f'{name}_both'].mean())
                        result['paired_95pct_vs_both_profiles'] = paired_date_interval(validation.date, references[f'{name}_both'], errors)
                    results.append(result)
                    print(f'{label} {name} {group}: MAE={result["mae"]:.4f}; delta={result["mae_change_vs_baseline"]:+.4f}', flush=True)
            predictions.to_csv(folder/f'predictions_{label}.csv', index=False)
    report = {'profile_timing': profile_timing, 'retrospective_lookahead': profile_timing == 'completed_current',
              'database_writes': False, 'production_models_overwritten': False, 'held_out_2025_26_read_or_scored': False,
              'design': 'Player proportions and opponent PPP relative to the possession-weighted league mean for each type. Include a both-profiles additive control before interactions. Imputation learns training medians only; unavailable player profiles remain flagged rather than dropping rookies. Fixed model settings, identical appearances across variants, paired date-block bootstrap intervals.',
              'split_adjustment': 'No 2021-22 profiles exist, so 2022-23 cannot train meaningful lagged profile effects. Use early eligible 2023-24 to train and validate from 2024-02-01, then train eligible 2023-24 and validate 2024-25. Retrain all baselines on those identical splits; not directly comparable with prior experiments.',
              'limitations': ['Previous-season profiles may be stale after trades, injuries, or role changes.',
                              'Unreported playtypes within available player profiles are zero-filled; entirely absent profiles are imputed with a missing indicator.',
                              'Synergy source coverage is not necessarily every regular-season game.',
                              'Development comparisons have been explored already; bootstrap intervals do not adjust for selection or multiple comparisons.',
                              'Average-based direction accuracy is uncalibrated and is not a sportsbook-line result.'],
              'coverage': coverage, 'results': results}
    if profile_timing == 'completed_current':
        report['interpretation'] = 'RETROSPECTIVE DIAGNOSTIC ONLY: full current-season profiles contain the scored game and future games. Scores do not estimate attainable pregame performance or justify deploying a profile-timing strategy.'
        report['split_adjustment'] = 'Use the exact same folds as the previous-season experiment solely for a controlled descriptive comparison. Baseline predictions and evaluation cohorts must match.'
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in x.items() if k != 'features'} for x in results]).to_csv(folder/'results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-snapshot', required=True, type=Path)
    parser.add_argument('--profile-snapshot', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--profile-timing', choices=['previous', 'completed_current'], default='previous')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.profile_timing == 'completed_current':
        print('RETROSPECTIVE ONLY: completed current-season profiles include future information; not a pregame backtest.', flush=True)
    profiles = json.loads(args.profile_snapshot.read_text()) if args.profile_snapshot else capture_profiles(args.profile_timing)
    profile_path = args.output_dir/'profile_snapshot.json'
    profile_path.write_text(json.dumps(profiles))
    run(json.loads(args.game_snapshot.read_text()), profiles, args.output_dir, args.profile_timing)
    report_path = args.output_dir/'report.json'
    report = json.loads(report_path.read_text())
    report.update({'game_snapshot': str(args.game_snapshot),
                   'game_snapshot_sha256': hashlib.sha256(args.game_snapshot.read_bytes()).hexdigest(),
                   'profile_snapshot_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest()})
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
