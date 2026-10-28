"""Test minutes-times-rate predictions on saved chronological development folds.

No database access or 2025-26 data. Score identical appearances against the saved
direct models. Separate ablations identify whether minutes or rate models help.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from models.experiment_features import SEASONS, metrics, paired_date_interval
from models.features.engineering import engineer_features, eligible_rows, get_feature_cols
from models.features.matchup import add_matchup_features
from models.features.offense import add_offense_features, offense_groups

KEYS = ['playerid', 'season', 'date', 'team', 'opponent']
MINUTE_FEATURES = ['minutes_avg5', 'minutes_avg10', 'minutes_avg_season',
                  'minutes_std10', 'minutes_trend', 'days_rest', 'is_back_to_back',
                  'is_home', 'fga_team_share_prior10', 'fga_team_share_trend',
                  'fga_prior5', 'fga_trend']


def estimator(name):
    return LinearRegression() if name == 'linear' else XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, random_state=42, n_jobs=1)


def production(minutes, rates):
    minutes, rates = np.asarray(minutes, dtype=float), np.asarray(rates, dtype=float)
    if minutes.shape != rates.shape or not np.isfinite(minutes).all() or not np.isfinite(rates).all():
        raise ValueError('Minutes and rates must be matching finite vectors')
    return np.maximum(0, minutes) * np.maximum(0, rates)


def rate_training_rows(train, stat):
    """Current minutes form training labels/weights only, never prediction X."""
    usable = train.loc[train.minutes > 0].copy()
    target = usable[stat].to_numpy() / usable.minutes.to_numpy()
    weights = usable.minutes.to_numpy() / usable.minutes.mean()
    if usable.empty or not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ValueError('Invalid per-minute training labels')
    return usable, target, weights


def direct_predictions(experiment, season, stat, validation):
    saved = pd.read_csv(experiment/f'predictions_{season}_{stat}.csv')
    saved.date = pd.to_datetime(saved.date)
    if len(saved) != len(validation):
        raise ValueError('Minutes experiment changed the validation cohort')
    joined = validation[KEYS + [stat]].merge(saved, on=KEYS, how='left',
                                            validate='one_to_one', suffixes=('', '_saved'), indicator=True)
    if not joined['_merge'].eq('both').all() or not np.array_equal(joined[stat], joined[f'{stat}_saved']):
        raise ValueError('Saved direct predictions differ from validation inputs')
    return {name: joined[f'{name}_shot_volume'].to_numpy() for name in ['linear', 'xgboost']}


def run(experiment, folder):
    source = experiment/'source_snapshot.json'
    snapshot = json.loads(source.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if json.loads((experiment/'report.json').read_text())['source_sha256'] != source_hash:
        raise ValueError('Original experiment snapshot checksum changed')
    players, teams = pd.DataFrame(snapshot['players']), pd.DataFrame(snapshot['teams'])
    if set(players.season)-set(SEASONS) or set(teams.season)-set(SEASONS):
        raise ValueError('Only pre-2025-26 development seasons may be read')
    with redirect_stdout(io.StringIO()):
        features = add_offense_features(add_matchup_features(engineer_features(players), teams))
    eligible = eligible_rows(features)
    minute_results, stat_results, coverage = [], [], []
    with threadpool_limits(limits=1):
        for season in ['2023-24', '2024-25']:
            train_seasons = [s for s in SEASONS if s < season]
            all_cols = list(dict.fromkeys(MINUTE_FEATURES + ['minutes'] +
                [c for s in ['points', 'rebounds', 'assists'] for c in get_feature_cols(s) +
                 offense_groups(s)['shot_volume'] + [f'{s}_per_minute_season', f'{s}_per_minute_10', s]]))
            common = eligible.dropna(subset=all_cols)
            train = common[common.season.isin(train_seasons)]
            validation = common[common.season == season]
            if train.empty or validation.empty or train.date.max() >= validation.date.min():
                raise ValueError('Invalid chronological split')
            if not np.isfinite(common[all_cols].to_numpy(dtype=float)).all():
                raise ValueError('Nonfinite experiment features')
            if len(train) != int(eligible.season.isin(train_seasons).sum()) or len(validation) != int(eligible.season.eq(season).sum()):
                raise ValueError('Missing features changed the prior experiment population')
            coverage.append({'season': season, 'train_seasons': train_seasons,
                             'train_rows': len(train), 'validation_rows': len(validation),
                             'zero_minute_validation_rows_retained': int(validation.minutes.eq(0).sum())})
            minutes = {label: validation[column].to_numpy() for label, column in [
                ('average_5', 'minutes_avg5'), ('average_10', 'minutes_avg10'),
                ('season_average', 'minutes_avg_season')]}
            minute_output = validation[KEYS + ['minutes']].copy()
            for name in ['linear', 'xgboost']:
                model = estimator(name)
                model.fit(train[MINUTE_FEATURES], train.minutes)
                minutes[name] = np.maximum(0, model.predict(validation[MINUTE_FEATURES]))
            actual_minutes = validation.minutes.to_numpy()
            ref_minute_errors = np.abs(actual_minutes-minutes['average_5'])
            for name, pred in minutes.items():
                result = {'season': season, 'variant': name, **metrics(actual_minutes, pred, validation.minutes_avg_season.to_numpy())}
                result['mae_change_vs_last5_minutes'] = float(np.abs(actual_minutes-pred).mean()-ref_minute_errors.mean())
                result['paired_date_95pct_mae_change'] = paired_date_interval(validation.date, ref_minute_errors, np.abs(actual_minutes-pred))
                minute_results.append(result)
                minute_output[name] = pred
                print(f'{season} minutes {name}: MAE={result["mae"]:.4f}', flush=True)
            minute_output.to_csv(folder/f'minutes_predictions_{season}.csv', index=False)
            for stat in ['points', 'rebounds', 'assists']:
                direct = direct_predictions(experiment, season, stat, validation)
                rate_features = get_feature_cols(stat) + offense_groups(stat)['shot_volume'] + [f'{stat}_per_minute_season', f'{stat}_per_minute_10']
                usable, rate_target, weights = rate_training_rows(train, stat)
                recent_rate = validation[f'{stat}_per_minute_10'].to_numpy()
                predictions = {f'direct_{name}': p for name, p in direct.items()}
                predictions['recent_minutes_recent_rate'] = production(minutes['average_5'], recent_rate)
                rate_forecasts = {}
                for name in ['linear', 'xgboost']:
                    # Give a direct model the union of inputs available to the
                    # two stages, separating architecture from extra information.
                    matched_columns = list(dict.fromkeys(MINUTE_FEATURES + rate_features))
                    direct_matched = estimator(name)
                    direct_matched.fit(train[matched_columns], train[stat])
                    predictions[f'direct_matched_{name}'] = direct_matched.predict(validation[matched_columns])
                    rate_model = estimator(name)
                    rate_model.fit(usable[rate_features], rate_target, sample_weight=weights)
                    rate_prediction = np.maximum(0, rate_model.predict(validation[rate_features]))
                    rate_forecasts[name] = rate_prediction
                    predictions[f'{name}_minutes_recent_rate'] = production(minutes[name], recent_rate)
                    predictions[f'recent_minutes_{name}_rate'] = production(minutes['average_5'], rate_prediction)
                    predictions[f'{name}_minutes_{name}_rate'] = production(minutes[name], rate_prediction)
                actual = validation[stat].to_numpy()
                reference_errors = {name: np.abs(actual-direct[name]) for name in direct}
                output = validation[KEYS + [stat]].copy()
                for name, pred in predictions.items():
                    errors = np.abs(actual-pred)
                    result = {'season': season, 'stat': stat, 'variant': name,
                              'rate_training_rows': len(usable), 'validation_rows': len(validation),
                              **metrics(actual, pred, validation[f'{stat}_avg_season'].to_numpy())}
                    for ref in ['linear', 'xgboost']:
                        result[f'mae_change_vs_direct_{ref}'] = float(errors.mean()-reference_errors[ref].mean())
                        result[f'paired_95pct_vs_direct_{ref}'] = paired_date_interval(validation.date, reference_errors[ref], errors)
                        matched_errors = np.abs(actual-predictions[f'direct_matched_{ref}'])
                        result[f'mae_change_vs_matched_direct_{ref}'] = float(errors.mean()-matched_errors.mean())
                        result[f'paired_95pct_vs_matched_direct_{ref}'] = paired_date_interval(validation.date, matched_errors, errors)
                    stat_results.append(result)
                    output[name] = pred
                    print(f'{season} {stat} {name}: MAE={result["mae"]:.4f}; vs direct linear={result["mae_change_vs_direct_linear"]:+.4f}', flush=True)
                output['recent_rate_per_minute'] = recent_rate
                for model_name, forecast in rate_forecasts.items():
                    output[f'{model_name}_rate_per_minute'] = forecast
                output.to_csv(folder/f'stat_predictions_{season}_{stat}.csv', index=False)
    report = {'database_writes': False, 'production_models_overwritten': False, 'held_out_2025_26_read_or_scored': False,
              'source_snapshot_sha256': source_hash, 'source_experiment': str(experiment),
              'minute_features': MINUTE_FEATURES,
              'rate_features': {s: get_feature_cols(s) + offense_groups(s)['shot_volume'] + [f'{s}_per_minute_season', f'{s}_per_minute_10'] for s in ['points', 'rebounds', 'assists']},
              'design': 'Fixed linear/XGBoost models trained on earlier seasons. Predict minutes and rates separately, multiply their nonnegative forecasts. Rate models use observed stats/minutes as training targets, weighting by minutes normalized to mean one. No actual validation minutes enter forecasts. Compare against exact saved direct shot-volume predictions, new direct models with the union of both stages input features, and recent-minutes/rate ablations on identical appearances.',
              'limitations': ['Observed appearances only: cannot evaluate DNP or availability forecasts.',
                              'No injury reports or prospective lineup inputs; role changes are inferred only from earlier games.',
                              'The first 20 season appearances and low-prior-minute players are outside this evaluation.',
                              'Development seasons have already been explored; this is not a fresh final holdout.',
                              'Product of expected minutes and expected rate need not equal expected total production.',
                              'Bootstrap intervals are exploratory, date-block paired, and not multiple-testing adjusted.'],
              'coverage': coverage, 'minutes': minute_results, 'stats': stat_results}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(minute_results).to_csv(folder/'minutes_results.csv', index=False)
    pd.DataFrame(stat_results).to_csv(folder/'stat_results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.experiment_dir, args.output_dir)


if __name__ == '__main__':
    main()
