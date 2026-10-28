"""Test known-absence scenarios with position weights and role adaptation."""

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

from models.experiment_features import SEASONS, metrics, paired_date_interval
from models.features.absences import absence_groups, add_absence_features
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.offense import add_offense_features, offense_groups
from nba.repair_shot_locations import connection


def capture():
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute('SELECT player_id,season,date,team,opponent,stint_number,stint_first_game_date,stint_last_game_date FROM public.player_game_absences WHERE season=ANY(%s) ORDER BY season,player_id,date', (SEASONS,))
            snapshot['absences'] = [dict(row) for row in cursor.fetchall()]
            cursor.execute('SELECT nba_api_id,position FROM public.players ORDER BY nba_api_id')
            snapshot['positions'] = [dict(row) for row in cursor.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=str))


def run(games, supplemental, folder):
    raw, absences = pd.DataFrame(games['players']), pd.DataFrame(supplemental['absences'])
    if set(raw.season) - set(SEASONS) or set(absences.season) - set(SEASONS):
        raise ValueError('Unexpected seasons')
    positions = pd.DataFrame(supplemental['positions'])
    with redirect_stdout(io.StringIO()):
        features = add_absence_features(add_offense_features(engineer_features(raw)), absences, positions)
    eligible = eligible_rows(features)
    results, coverage, slices = [], [], []
    with threadpool_limits(limits=1):
        for stat in ['points', 'rebounds', 'assists']:
            base = get_feature_cols(stat) + offense_groups(stat)['shot_volume']
            common = eligible.dropna(subset=base + [stat])
            for season in ['2023-24', '2024-25']:
                train = common[common.season < season]
                validation = common[common.season == season]
                if train.empty or validation.empty or train.date.max() >= validation.date.min():
                    raise ValueError('Invalid chronological split')
                coverage.append({'season': season, 'stat': stat, 'train_rows': len(train), 'validation_rows': len(validation),
                                 'rows_with_absences': int(validation.absent_count.gt(0).sum()),
                                 'rows_with_missing_rotation_minutes': int(validation.absent_minutes.ge(20).sum()),
                                 'rows_with_unknown_position_minutes': int(validation.absent_unknown_position_minutes.gt(0).sum()),
                                 'rows_with_thin_profiles': int(validation.absent_thin_profile_count.gt(0).sum())})
                actual, lines = validation[stat].to_numpy(), validation[f'{stat}_avg_season'].to_numpy()
                output = validation[['playerid', 'season', 'date', 'team', 'opponent', stat,
                                     *absence_groups()['position_and_adaptation']]].copy()
                references = {}
                for variant, extras in absence_groups().items():
                    columns = base + extras
                    if not np.isfinite(common[columns].to_numpy(float)).all():
                        raise ValueError('Nonfinite input')
                    for name in ['linear', 'xgboost']:
                        model = LinearRegression() if name == 'linear' else XGBRegressor(
                            n_estimators=200, max_depth=6, learning_rate=.05,
                            subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=1)
                        model.fit(train[columns], train[stat])
                        prediction = model.predict(validation[columns])
                        errors = np.abs(actual - prediction)
                        output[f'{name}_{variant}'] = prediction
                        references[(name, variant)] = errors
                        baseline = references[(name, 'baseline')]
                        row = {'season': season, 'stat': stat, 'model': name, 'variant': variant, 'features': columns,
                               'train_rows': len(train), 'validation_rows': len(validation),
                               **metrics(actual, prediction, lines),
                               'mae_change_vs_baseline': float(np.mean(errors - baseline)),
                               'paired_95pct_vs_baseline': paired_date_interval(validation.date, baseline, errors)}
                        control = {'position_absences': 'unweighted_absences',
                                   'position_and_adaptation': 'position_absences'}.get(variant)
                        if control:
                            reference = references[(name, control)]
                            row.update({'incremental_control': control,
                                        'incremental_mae_change': float(np.mean(errors - reference)),
                                        'incremental_paired_95pct': paired_date_interval(validation.date, reference, errors)})
                        results.append(row)
                        for label, mask in [('missing_20plus_minutes', validation.absent_minutes.ge(20).to_numpy()),
                                            ('recent_disruption', validation.new_absence_minutes.ge(10).to_numpy())]:
                            if mask.any():
                                slices.append({'season': season, 'stat': stat, 'model': name, 'variant': variant,
                                               'slice': label, 'rows': int(mask.sum()),
                                               **metrics(actual[mask], prediction[mask], lines[mask]),
                                               'mae_change_vs_baseline': float(np.mean(errors[mask] - baseline[mask])),
                                               'paired_95pct_vs_baseline': paired_date_interval(validation.loc[mask, 'date'], baseline[mask], errors[mask])})
                        print(f'{season} {stat} {name} {variant}: MAE={row["mae"]:.4f}; delta={row["mae_change_vs_baseline"]:+.4f}', flush=True)
                output.to_csv(folder/f'predictions_{season}_{stat}.csv', index=False)
    report = {'database_writes': False, 'production_models_overwritten': False, 'held_out_2025_26_read_or_scored': False,
              'evaluation': 'Conditional on actual historical absences, not an availability forecast or timestamped pregame backtest.',
              'design': {'profile': 'Last ten appearances for same team and stint before each absence episode, frozen through consecutive missed team games.',
                         'weights': 'identical parsed position sets=1, overlapping=.6, disjoint/unknown=.3; broad G/F labels expand to their two positions',
                         'usage': 'Box-score estimate: (FGA+.44*FTA+TOV)*(team minutes/5)/(player minutes*team used possessions), pooled over prior appearances; not play-by-play usage.',
                         'loads': 'Minutes, estimated used possessions, usage fraction times minutes, FGA, assists and rebounds; summed over absent teammates.',
                         'adaptation': 'Weighted load multiplied by 2**(-prior target appearances during that absence/5). Ongoing load remains alongside faded load.',
                         'splits': '2022-23 -> 2023-24; 2022-23/2023-24 -> 2024-25. Fixed OLS/XGBoost, identical appearances.'},
              'limitations': ['Absence table covers interior stint gaps only, as requested; boundary/transition gaps are omitted.',
                              'Absences are observed after games and stints use later appearances; this tests a supplied-known-absence scenario.',
                              'Positions come from current player metadata, not dated historical positions; missing positions use .3 with a missing-position feature.',
                              'Profiles with fewer than five prior appearances are counted explicitly; no outcome-minute filtering.',
                              'No clusters, projected lineups, rotation assignments, or return-from-absence effects in this first experiment.',
                              'Development folds have been explored; date-block intervals are exploratory without multiple-comparison correction.',
                              'Proxy direction scores use season-to-date averages, not real betting lines or calibrated confidence.'],
              'coverage': coverage, 'results': results, 'slices': slices}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results]).to_csv(folder/'results.csv', index=False)
    pd.DataFrame(slices).to_csv(folder/'slice_results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-snapshot', required=True, type=Path)
    parser.add_argument('--supplemental-snapshot', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    supplemental = json.loads(args.supplemental_snapshot.read_text()) if args.supplemental_snapshot else capture()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir/'supplemental_snapshot.json'
    source.write_text(json.dumps(supplemental, indent=2))
    run(json.loads(args.game_snapshot.read_text()), supplemental, args.output_dir)
    path = args.output_dir/'report.json'
    report = json.loads(path.read_text())
    report.update({'game_snapshot': str(args.game_snapshot),
                   'game_snapshot_sha256': hashlib.sha256(args.game_snapshot.read_bytes()).hexdigest(),
                   'supplemental_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'supplemental_source': 'live_database' if not args.supplemental_snapshot else str(args.supplemental_snapshot)})
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
