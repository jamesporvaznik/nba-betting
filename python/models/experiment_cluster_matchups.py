"""Read-only cluster capture and chronological points/rebounds/assists tests."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
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
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.experiment_features import metrics, paired_date_interval, SEASONS
from models.experiment_playtypes import chronological_folds
from models.features.cluster_matchup import add_cluster_matchups, cluster_groups, STATS
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.offense import add_offense_features, offense_groups
from nba.repair_shot_locations import connection


def profile_seasons(profile_timing):
    return ['2022-23', '2023-24'] if profile_timing == 'previous' else ['2023-24', '2024-25']


def capture_clusters(profile_timing='previous'):
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(),
                'source': 'public.player_clusters', 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute('SELECT player_id,season,primary_archetype FROM public.player_clusters WHERE season=ANY(%s) ORDER BY season,player_id',
                           (profile_seasons(profile_timing),))
            snapshot['assignments'] = [dict(row) for row in cursor.fetchall()]
        conn.rollback()
    return snapshot


def run(snapshot, assignments, folder, profile_timing='previous'):
    raw = pd.DataFrame(snapshot['players'])
    if set(raw.season) - set(SEASONS) or set(assignments.season) - set(profile_seasons(profile_timing)):
        raise ValueError('Unexpected game/profile seasons')
    with redirect_stdout(io.StringIO()):
        features = add_cluster_matchups(add_offense_features(engineer_features(raw)), assignments, profile_timing=profile_timing)
    prefix = 'retrospective' if profile_timing == 'completed_current' else 'lag'
    available, cluster_label = f'{prefix}_cluster_available', f'{prefix}_cluster'
    eligible = eligible_rows(features)
    results, coverage, covered_results = [], [], []
    with threadpool_limits(limits=1):
        for stat in STATS:
            base = get_feature_cols(stat) + offense_groups(stat)['shot_volume']
            common = eligible.dropna(subset=base + [stat])
            for label, train, validation in chronological_folds(common):
                if train.empty or validation.empty or train.date.max() >= validation.date.min():
                    raise ValueError('Invalid chronological split')
                coverage.append({'fold': label, 'stat': stat, 'train_rows': len(train),
                                 'validation_rows': len(validation),
                                 'train_last_date': str(train.date.max().date()),
                                 'validation_first_date': str(validation.date.min().date()),
                                 'validation_cluster_missing': int(validation[available].eq(0).sum()),
                                 'validation_no_peer_history': int(validation.cluster_decayed_appearances.eq(0).sum()),
                                 'median_decayed_peer_appearances': float(validation.cluster_decayed_appearances.median())})
                actual = validation[stat].to_numpy()
                # Retain the prior experiment's proxy for comparable diagnostics;
                # rolling baselines define matchup features, not betting lines.
                lines = validation[f'{stat}_avg_season'].to_numpy()
                extra_cols = list(dict.fromkeys(c for cols in cluster_groups(stat, profile_timing).values() for c in cols))
                output = validation[['playerid', 'season', 'date', 'team', 'opponent', stat,
                                     cluster_label, *extra_cols]].copy()
                references = {}
                for variant, extras in cluster_groups(stat, profile_timing).items():
                    columns = base + extras
                    if not np.isfinite(common[columns].to_numpy(dtype=float)).all():
                        raise ValueError('Invalid features')
                    for name in ['linear', 'xgboost']:
                        model = LinearRegression() if name == 'linear' else XGBRegressor(
                            n_estimators=200, max_depth=6, learning_rate=.05,
                            subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=1)
                        model.fit(train[columns], train[stat])
                        prediction = model.predict(validation[columns])
                        errors = np.abs(actual - prediction)
                        output[f'{name}_{variant}'] = prediction
                        if variant == 'baseline':
                            references[name] = errors
                        if variant == 'opponent_rolling10':
                            references[f'{name}_opponent'] = errors
                        if variant == 'cluster_availability':
                            references[f'{name}_availability'] = errors
                        row = {'fold': label, 'stat': stat, 'model': name, 'variant': variant,
                               'features': columns, 'train_rows': len(train), 'validation_rows': len(validation),
                               **metrics(actual, prediction, lines),
                               'mae_change_vs_baseline': float(errors.mean() - references[name].mean()),
                               'paired_95pct_vs_baseline': paired_date_interval(validation.date, references[name], errors)}
                        if variant == 'opponent_and_cluster_rolling10':
                            reference = references[f'{name}_opponent']
                            row.update({'mae_change_vs_opponent': float(errors.mean() - reference.mean()),
                                        'paired_95pct_vs_opponent': paired_date_interval(validation.date, reference, errors)})
                        if variant == 'cluster_rolling10':
                            reference = references[f'{name}_availability']
                            row.update({'mae_change_vs_availability': float(errors.mean() - reference.mean()),
                                        'paired_95pct_vs_availability': paired_date_interval(validation.date, reference, errors)})
                        results.append(row)
                        covered = validation[available].eq(1).to_numpy()
                        if covered.any():
                            covered_results.append({'fold': label, 'stat': stat, 'model': name, 'variant': variant,
                                                    'rows': int(covered.sum()),
                                                    **metrics(actual[covered], prediction[covered], lines[covered]),
                                                    'mae_change_vs_baseline': float(np.mean(errors[covered] - references[name][covered])),
                                                    'paired_95pct_vs_baseline': paired_date_interval(validation.loc[covered, 'date'], references[name][covered], errors[covered])})
                        print(f'{label} {stat} {name} {variant}: MAE={row["mae"]:.4f}; delta={row["mae_change_vs_baseline"]:+.4f}', flush=True)
                output.to_csv(folder / f'predictions_{label}_{stat}.csv', index=False)
    report = {'database_writes': False, 'production_models_overwritten': False,
              'profile_timing': profile_timing, 'retrospective_lookahead': profile_timing == 'completed_current',
              'game_seasons': SEASONS, 'cluster_source_seasons': profile_seasons(profile_timing),
              'held_out_2025_26_games_read_or_scored': False,
              'design': {'rolling_windows': [5, 10], 'half_life_days': 30, 'shrinkage_pseudo_appearances': 20,
                         'peer_source_min_prior_games': 10, 'peer_source_min_prior10_avg_minutes': 10,
                         'raw_residual': 'actual stat minus prior-window mean',
                         'minutes_residual': 'actual stat minus actual minutes times prior-window sum(stat)/sum(minutes)',
                         'aggregation': 'sum(decayed weight * residual)/(sum(decayed weight) + 20)',
                         'peer_selection': f'same primary cluster, timing={profile_timing}; exclude target player',
                         'timing': 'strictly earlier dates within current season; emit whole date before updating histories',
                         'missing': 'retain player, zero cluster effects/count with availability flag',
                         'comparison': 'fixed baseline + shot volume, cluster-availability-only control, all-player opponent effects, cluster effects, both; 5-game sensitivity'},
              'limitations': [('Completed-current-season clusters use future information: retrospective diagnostic, not attainable pregame validation.'
                               if profile_timing == 'completed_current' else 'Previous-season clusters may be stale; these results do not test dated current-season clusters.'),
                              'Cluster labels only select within-season peers and are never numeric model inputs.',
                              'Primary clusters only; no continuous style-distance or soft-membership comparison in this experiment.',
                              'Decayed appearance mass is not an independent-game effective sample size.',
                              'Minutes residuals use actual minutes only for completed past games; target minutes never enter prediction features.',
                              'Validation history updates after each completed date; fitted models stay fixed.',
                              'Development folds have been explored before; date-block intervals are exploratory and not multiplicity-adjusted.',
                              'Proxy direction accuracy uses prior season-to-date averages, not sportsbook lines or calibrated probabilities.',
                              'No lineup/availability information; no database or saved production-model changes.'],
              'coverage': coverage, 'results': results,
              'covered_players_results': covered_results}
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results]).to_csv(folder / 'results.csv', index=False)
    pd.DataFrame(covered_results).to_csv(folder / 'covered_players_results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-snapshot', type=Path, required=True)
    parser.add_argument('--cluster-snapshot', type=Path,
                        help='Optional replay of an experiment snapshot; default reads live player_clusters')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile-timing', choices=['previous', 'completed_current'], default='previous')
    args = parser.parse_args()
    snapshot = json.loads(args.cluster_snapshot.read_text()) if args.cluster_snapshot else capture_clusters(args.profile_timing)
    assignments = pd.DataFrame(snapshot['assignments'])
    if set(assignments.season) != set(profile_seasons(args.profile_timing)):
        raise ValueError('Both required cluster seasons must exist')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cluster_path = args.output_dir / 'cluster_snapshot.json'
    cluster_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print('Using cached game logs and live database cluster assignments.' if not args.cluster_snapshot else 'Replaying experiment inputs.', flush=True)
    run(json.loads(args.game_snapshot.read_text()), assignments, args.output_dir, args.profile_timing)
    path = args.output_dir / 'report.json'
    report = json.loads(path.read_text())
    report.update({'game_snapshot': str(args.game_snapshot),
                   'game_snapshot_sha256': hashlib.sha256(args.game_snapshot.read_bytes()).hexdigest(),
                   'cluster_source': 'live_database' if not args.cluster_snapshot else str(args.cluster_snapshot),
                   'cluster_snapshot_sha256': hashlib.sha256(cluster_path.read_bytes()).hexdigest()})
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
