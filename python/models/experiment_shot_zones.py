"""Read-only points experiments using lagged player/opponent shot-zone profiles.

Use the exact folds from the playtype experiment: early 2023-24 -> late 2023-24,
then 2023-24 -> 2024-25. No current-season profile or 2025-26 model input.
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
from models.experiment_playtypes import chronological_folds
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.offense import add_offense_features, offense_groups
from models.features.shot_zone_matchup import add_shot_zone_matchups, shot_zone_columns
from nba.repair_shot_locations import connection


def capture_profiles():
    queries = {
        'players': 'SELECT player_id,season,shot_zone,frequency_pct FROM public.player_shot_locations WHERE season = ANY(%s) ORDER BY season,player_id,shot_zone',
        'defense': 'SELECT t.abbreviation AS opponent,s.season,s.shot_zone,s.fg_pct_allowed,s.frequency_pct FROM public.team_defensive_shot_locations s JOIN public.teams t ON t.nba_api_id=s.team_id WHERE s.season = ANY(%s) ORDER BY s.season,t.abbreviation,s.shot_zone',
    }
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            for name, query in queries.items():
                c.execute(query, (['2022-23', '2023-24'],))
                snapshot[name] = [dict(r) for r in c.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def run(game_snapshot, profiles, folder):
    raw = pd.DataFrame(game_snapshot['players'])
    if set(raw.season)-set(SEASONS):
        raise ValueError('No 2025-26 or other game seasons allowed')
    for records in [profiles['players'], profiles['defense']]:
        if {r['season'] for r in records} - {'2022-23', '2023-24'}:
            raise ValueError('Only completed 2022-23/2023-24 shot profiles are allowed')
    with redirect_stdout(io.StringIO()):
        features = add_offense_features(engineer_features(raw))
        features = add_shot_zone_matchups(features, pd.DataFrame(profiles['players']), pd.DataFrame(profiles['defense']))
    base = get_feature_cols('points') + offense_groups('points')['shot_volume']
    eligible = eligible_rows(features).dropna(subset=base + ['points'])
    if not np.isfinite(eligible[base + ['points']].to_numpy(dtype=float)).all():
        raise ValueError('Invalid baseline features')
    groups, results, coverage = shot_zone_columns(), [], []
    with threadpool_limits(limits=1):
        for label, train, validation in chronological_folds(eligible):
            if train.empty or validation.empty or train.date.max() >= validation.date.min():
                raise ValueError('Invalid chronological shot-zone split')
            coverage.append({'fold': label, 'train_rows': len(train), 'validation_rows': len(validation),
                             'train_first_date': str(train.date.min().date()), 'train_last_date': str(train.date.max().date()),
                             'validation_first_date': str(validation.date.min().date()), 'validation_last_date': str(validation.date.max().date()),
                             'train_player_profile_missing': int(train.lag_zone_player_available.eq(0).sum()),
                             'validation_player_profile_missing': int(validation.lag_zone_player_available.eq(0).sum()),
                             'validation_opponent_profile_missing': int(validation.lag_zone_defense_available.eq(0).sum())})
            actual, lines = validation.points.to_numpy(), validation.points_avg_season.to_numpy()
            output = validation[['playerid', 'season', 'date', 'team', 'opponent', 'points',
                                 'lag_zone_player_available', 'lag_zone_defense_available']].copy()
            references = {}
            for variant, extras in groups.items():
                columns = base + extras
                for name in ['linear', 'xgboost']:
                    model = LinearRegression() if name == 'linear' else XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
                        colsample_bytree=.8, random_state=42, n_jobs=1)
                    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)
                    pipeline.fit(train[columns], train.points)
                    prediction = pipeline.predict(validation[columns])
                    errors = np.abs(actual-prediction)
                    output[f'{name}_{variant}'] = prediction
                    result = {'fold': label, 'model': name, 'variant': variant, 'features': columns,
                              'train_rows': len(train), 'validation_rows': len(validation), **metrics(actual, prediction, lines)}
                    if variant == 'baseline':
                        references[name] = errors
                    result['mae_change_vs_baseline'] = float(errors.mean()-references[name].mean())
                    result['paired_95pct_vs_baseline'] = paired_date_interval(validation.date, references[name], errors)
                    if variant == 'both_profiles':
                        references[f'{name}_both'] = errors
                    if variant == 'interactions':
                        result['mae_change_vs_both_profiles'] = float(errors.mean()-references[f'{name}_both'].mean())
                        result['paired_95pct_vs_both_profiles'] = paired_date_interval(validation.date, references[f'{name}_both'], errors)
                    results.append(result)
                    print(f'{label} {name} {variant}: MAE={result["mae"]:.4f}; delta={result["mae_change_vs_baseline"]:+.4f}', flush=True)
            output.to_csv(folder/f'predictions_{label}.csv', index=False)
    report = {'database_writes': False, 'production_models_overwritten': False, 'held_out_2025_26_read_or_scored': False,
              'profile_timing': 'previous_season_only',
              'design': 'Same points baseline, folds and model parameters as lagged playtype experiment. Compare six-zone player frequency, opponent zone FG% and frequency relative to the average team, additive profiles, and interactions. Scoring interactions weight player zone share by opponent FG% gap and two/three-point value; frequency interactions measure shot-distribution alignment. Imputation uses training medians only and explicit missing-profile flags; no players removed for profile availability.',
              'split_adjustment': 'No 2021-22 profiles: train eligible 2023-24 through Jan 31 and validate Feb onward; then train eligible 2023-24 and validate 2024-25. Every variant uses identical appearances.',
              'limitations': ['Previous-season profiles can miss current roles and roster changes.',
                              'Missing zones within an available player profile become zero; entirely missing profiles remain flagged and imputed.',
                              'League centering uses mean team rates, not a pooled shot-weighted league FG%.',
                              'Six-zone profiles exclude backcourt shots and do not model free throws.',
                              'Interactions are matchup proxies, not independently validated scoring projections.',
                              'Development comparisons and date-block intervals are exploratory, without multiple-comparison correction.',
                              'Direction scores are uncalibrated average-proxy results, not sportsbook accuracy.'],
              'coverage': coverage, 'results': results}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in x.items() if k != 'features'} for x in results]).to_csv(folder/'results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-snapshot', type=Path, required=True)
    parser.add_argument('--profile-snapshot', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    profiles = json.loads(args.profile_snapshot.read_text()) if args.profile_snapshot else capture_profiles()
    profile_path = args.output_dir/'profile_snapshot.json'
    profile_path.write_text(json.dumps(profiles))
    run(json.loads(args.game_snapshot.read_text()), profiles, args.output_dir)
    report_path = args.output_dir/'report.json'
    report = json.loads(report_path.read_text())
    report.update({'game_snapshot': str(args.game_snapshot),
                   'game_snapshot_sha256': hashlib.sha256(args.game_snapshot.read_bytes()).hexdigest(),
                   'profile_snapshot_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest()})
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
