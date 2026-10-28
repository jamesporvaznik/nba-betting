"""Test team rebounding and lagged player tracking against the rebounds baseline."""

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
from models.features.rebounding import add_rebounding_features, rebound_columns, TRACKING
from nba.repair_shot_locations import connection


def capture():
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute('SELECT season,date,team,opponent_abbrev,team_orb,team_trb,opponent_orb,opponent_trb,opponent_fga,opponent_fgm FROM public.team_defensive_stats WHERE season=ANY(%s) ORDER BY season,team,date', (SEASONS,))
            snapshot['teams'] = [dict(r) for r in c.fetchall()]
            c.execute('SELECT player_id,season,reb_chance_pct,dreb_chance_pct,oreb_chance_pct,reb_contest_pct FROM public.player_tracking_stats WHERE season=ANY(%s) ORDER BY season,player_id', (['2022-23', '2023-24'],))
            snapshot['tracking'] = [dict(r) for r in c.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def run(games, supplemental, folder):
    raw, teams, tracking = pd.DataFrame(games['players']), pd.DataFrame(supplemental['teams']), pd.DataFrame(supplemental['tracking'])
    if set(raw.season)-set(SEASONS) or set(teams.season)-set(SEASONS) or set(tracking.season)-{'2022-23', '2023-24'}:
        raise ValueError('Unexpected input seasons; no 2025-26 or current-year tracking allowed')
    with redirect_stdout(io.StringIO()):
        features = add_offense_features(engineer_features(raw))
        features = add_rebounding_features(features, teams, tracking)
    base = get_feature_cols('rebounds') + offense_groups('rebounds')['shot_volume']
    eligible = eligible_rows(features).dropna(subset=base + ['rebounds'])
    if not np.isfinite(eligible[base + ['rebounds']].to_numpy(dtype=float)).all():
        raise ValueError('Invalid baseline inputs')
    results, coverage = [], []
    folds = [(label, train, validation, rebound_columns()) for label, train, validation in chronological_folds(eligible)]
    for season in ['2023-24', '2024-25']:
        earlier = [s for s in SEASONS if s < season]
        folds.append((f'{season}_team_only_full_training', eligible[eligible.season.isin(earlier)],
                      eligible[eligible.season == season],
                      {k: v for k, v in rebound_columns().items() if k in ['baseline', 'team_rebounding']}))
    with threadpool_limits(limits=1):
        for label, train, validation, variants in folds:
            if train.empty or validation.empty or train.date.max() >= validation.date.min():
                raise ValueError('Invalid chronological rebounding split')
            coverage.append({'fold': label, 'train_rows': len(train), 'validation_rows': len(validation),
                             'variants': list(variants),
                             'train_tracking_missing': int(train.lag_reb_tracking_available.eq(0).sum()),
                             'validation_tracking_missing': int(validation.lag_reb_tracking_available.eq(0).sum()),
                             'validation_team_history_missing': int(validation[rebound_columns()['team_rebounding']].isna().any(axis=1).sum())})
            actual, lines = validation.rebounds.to_numpy(), validation.rebounds_avg_season.to_numpy()
            output = validation[['playerid', 'season', 'date', 'team', 'opponent', 'rebounds', 'lag_reb_tracking_available']].copy()
            references = {}
            for variant, extra in variants.items():
                columns = base + extra
                for name in ['linear', 'xgboost']:
                    model = LinearRegression() if name == 'linear' else XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
                        colsample_bytree=.8, random_state=42, n_jobs=1)
                    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)
                    pipeline.fit(train[columns], train.rebounds)
                    prediction = pipeline.predict(validation[columns])
                    errors = np.abs(actual-prediction)
                    output[f'{name}_{variant}'] = prediction
                    if variant == 'baseline':
                        references[name] = errors
                    if variant == 'both':
                        references[f'{name}_both'] = errors
                    row = {'fold': label, 'model': name, 'variant': variant, 'features': columns,
                           'train_rows': len(train), 'validation_rows': len(validation),
                           **metrics(actual, prediction, lines),
                           'mae_change_vs_baseline': float(errors.mean()-references[name].mean()),
                           'paired_95pct_vs_baseline': paired_date_interval(validation.date, references[name], errors)}
                    if variant == 'interactions':
                        row['mae_change_vs_both'] = float(errors.mean()-references[f'{name}_both'].mean())
                        row['paired_95pct_vs_both'] = paired_date_interval(validation.date, references[f'{name}_both'], errors)
                    results.append(row)
                    print(f'{label} {name} {variant}: MAE={row["mae"]:.4f}; delta={row["mae_change_vs_baseline"]:+.4f}', flush=True)
            output.to_csv(folder/f'predictions_{label}.csv', index=False)
    report = {'database_writes': False, 'production_models_overwritten': False, 'held_out_2025_26_read_or_scored': False,
              'tracking_table': 'player_tracking_stats', 'tracking_fields': TRACKING,
              'design': 'Baseline, prior-game team rebounding, previous-season player tracking, both, and two tracking-by-opponent interactions. Identical appearances; fixed models and train-only imputation. Team opponent_orb is offensive boards conceded by that team: joining the upcoming opponent history measures offensive boards it has allowed. Own/opponent defensive boards derive as TRB minus ORB. FG misses use reciprocal team-game rows. Rates are ratios of prior cumulative or ten-game rebound counts.',
              'splits': 'Lagged tracking comparisons: early 2023-24 -> Feb onward; 2023-24 -> 2024-25. Additional team-only controls use full expanding folds: 2022-23 -> 2023-24, and 2022-23/2023-24 -> 2024-25. No tracking features enter the additional controls.',
              'limitations': ['Player tracking profiles are completed previous-season rates, not current-season snapshots or raw rebound chances.',
                              'No player_tracking_data table exists; the four rebounding fields are in player_tracking_stats.',
                              'Team rebounding history is current season before game, including all stored game types.',
                              'Matchup products are fitted interactions, not direct forecasts of individual rebound chances.',
                              'No prospective availability/lineup information. Development scores are not fresh holdout evidence.',
                              'Date-block bootstrap intervals are exploratory and not multiple-testing adjusted.',
                              'Average-proxy direction scores are uncalibrated, not sportsbook accuracy.'],
              'coverage': coverage, 'results': results}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results]).to_csv(folder/'results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-snapshot', required=True, type=Path)
    parser.add_argument('--supplemental-snapshot', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    supplemental = json.loads(args.supplemental_snapshot.read_text()) if args.supplemental_snapshot else capture()
    path = args.output_dir/'supplemental_snapshot.json'
    path.write_text(json.dumps(supplemental))
    run(json.loads(args.game_snapshot.read_text()), supplemental, args.output_dir)
    report_path = args.output_dir/'report.json'
    report = json.loads(report_path.read_text())
    report.update({'game_snapshot': str(args.game_snapshot), 'game_snapshot_sha256': hashlib.sha256(args.game_snapshot.read_bytes()).hexdigest(),
                   'supplemental_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
