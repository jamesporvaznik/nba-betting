"""Offline-reproducible, read-only chronological feature-group experiments.

Use 2023-24 and 2024-25 validation folds. Never read or score 2025-26.
Keep parameters fixed; compare every variant on identical eligible rows.
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
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.matchup import add_matchup_features, experimental_groups
from models.features.offense import add_offense_features, offense_groups
from nba.repair_shot_locations import connection

SEASONS = ['2022-23', '2023-24', '2024-25']


from models.evaluation.metrics import metrics, paired_date_interval

def capture(suite='matchup'):
    queries = {
        'players': 'SELECT playerid,gameid,season,date,team,opponent,homeaway,minutes,points,rebounds,assists,steals,blocks,turnovers FROM public.player_gamelog WHERE season = ANY(%s) ORDER BY playerid,date',
        'teams': 'SELECT season,date,team,team_pace,def_rating,opponent_fgm,opponent_fga,opponent_3pm,opponent_3pa,opponent_fta,opponent_pts,opponent_trb,opponent_ast FROM public.team_defensive_stats WHERE season = ANY(%s) ORDER BY season,team,date',
    }
    if suite == 'offense':
        queries['players'] = queries['players'].replace('turnovers FROM', 'turnovers,fga,fta,"3PA" FROM')
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            for name, query in queries.items():
                c.execute(query, (SEASONS,))
                snapshot[name] = [dict(r) for r in c.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))






def run(snapshot, folder, suite='matchup'):
    player_rows = pd.DataFrame(snapshot['players'])
    teams = pd.DataFrame(snapshot['teams'])
    if set(player_rows.season) - set(SEASONS) or set(teams.season) - set(SEASONS):
        raise ValueError('Experiment must not contain held-out 2025-26 data or other seasons')
    with redirect_stdout(io.StringIO()):
        features = add_matchup_features(engineer_features(player_rows), teams)
        if suite == 'offense':
            features = add_offense_features(features)
    group_builder = offense_groups if suite == 'offense' else experimental_groups
    eligible = eligible_rows(features)
    results = []
    coverage = []
    with threadpool_limits(limits=1):
        for validation in ['2023-24', '2024-25']:
            train_seasons = [s for s in SEASONS if s < validation]
            for stat in ['points', 'rebounds', 'assists']:
                groups = group_builder(stat)
                common_columns = list(dict.fromkeys(get_feature_cols(stat) + groups['combined'] + [stat]))
                common = eligible.dropna(subset=common_columns)
                tr = common[common.season.isin(train_seasons)]
                va = common[common.season == validation]
                if tr.empty or va.empty or tr.date.max() >= va.date.min():
                    raise ValueError('Invalid chronological experiment split')
                if not np.isfinite(common[common_columns].to_numpy(dtype=float)).all():
                    raise ValueError('Nonfinite experiment input')
                coverage.append({'validation_season': validation, 'stat': stat, 'training_seasons': train_seasons,
                                 'eligible_train': int(eligible.season.isin(train_seasons).sum()),
                                 'eligible_validation': int((eligible.season == validation).sum()),
                                 'common_train': len(tr), 'common_validation': len(va)})
                actual = va[stat].to_numpy()
                lines = va[f'{stat}_avg_season'].to_numpy()
                predictions = va[['playerid', 'season', 'date', 'team', 'opponent', stat]].copy()
                baseline_errors = {}
                baseline_mae = {}
                for group, extras in groups.items():
                    columns = get_feature_cols(stat) + extras
                    for model_name in ['linear', 'xgboost']:
                        model = LinearRegression() if model_name == 'linear' else XGBRegressor(
                            n_estimators=200, max_depth=6, learning_rate=.05,
                            subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=1)
                        model.fit(tr[columns], tr[stat])
                        pred = model.predict(va[columns])
                        predictions[f'{model_name}_{group}'] = pred
                        errors = np.abs(actual-pred)
                        scores = metrics(actual, pred, lines)
                        if group == 'baseline':
                            baseline_errors[model_name] = errors
                            baseline_mae[model_name] = scores['mae']
                        scores.update({'validation_season': validation, 'stat': stat, 'model': model_name,
                                       'group': group, 'features': columns, 'train_rows': len(tr), 'validation_rows': len(va),
                                       'mae_change_vs_same_model_baseline': scores['mae']-baseline_mae[model_name],
                                       'paired_date_bootstrap_95pct_mae_change': paired_date_interval(va.date, baseline_errors[model_name], errors)})
                        results.append(scores)
                        print(f'{validation} {stat} {model_name} {group}: MAE {scores["mae"]:.4f}; delta {scores["mae_change_vs_same_model_baseline"]:+.4f}', flush=True)
                predictions.to_csv(folder / f'predictions_{validation}_{stat}.csv', index=False)
    report = {'suite': suite, 'database_writes': False, 'held_out_2025_26_read_or_scored': False,
              'design': 'Fixed baseline parameters, same eligible cohort per fold/stat across all variants. Train only seasons before each validation season. Paired date-block bootstrap is exploratory; it does not correct for multiple comparisons or model-selection uncertainty. No saved production models overwritten.',
              'features': {s: group_builder(s) for s in ['points', 'rebounds', 'assists']},
              'coverage': coverage, 'results': results}
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results]).to_csv(folder / 'results.csv', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--suite', choices=['matchup', 'offense'], default='matchup')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else capture(args.suite)
    path = args.output_dir / 'source_snapshot.json'
    path.write_text(json.dumps(snapshot))
    print('Captured/reused read-only 2022-23 through 2024-25 inputs; no 2025-26 data.', flush=True)
    report = run(snapshot, args.output_dir, args.suite)
    report['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Experiment complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
