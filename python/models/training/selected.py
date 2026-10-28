"""Train the fixed selected blend and calibrate using earlier-season predictions."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models.data.load import TRAINING_SEASONS, TEST_SEASON
from models.data.snapshot import restrict_snapshot
from models.evaluation.confidence import calibration_fit, assess, KEYS
from models.features.engineering import eligible_rows
from models.features.selected import build_selected_features, selected_feature_cols, STATS
from models.training.blend import SelectedBlendRegressor


def calibration_folds(training):
    for season in TRAINING_SEASONS[1:]:
        earlier = training[training.season.isin([s for s in TRAINING_SEASONS if s < season])]
        later = training[training.season.eq(season)]
        if earlier.empty or later.empty or earlier.date.max() >= later.date.min():
            raise ValueError(f'Insufficient chronological calibration data for {season}')
        yield earlier, later


def train_selected(snapshot, stats=None, min_prior_games=20, min_prior_avg_minutes=10., evaluate=False):
    stats = STATS if stats is None else list(stats)
    if not stats or set(stats)-set(STATS) or len(stats) != len(set(stats)):
        raise ValueError('Choose unique targets from points, rebounds, assists')
    seasons = TRAINING_SEASONS + ([TEST_SEASON] if evaluate else [])
    snapshot = restrict_snapshot(snapshot, seasons)
    print('Building selected features from complete histories, then applying pregame eligibility.', flush=True)
    frame = eligible_rows(build_selected_features(snapshot), min_prior_games, min_prior_avg_minutes)
    results, predictions, calibration_outputs = {}, {}, {}
    with threadpool_limits(limits=1):
        for stat in stats:
            columns = selected_feature_cols(stat)
            required = list(dict.fromkeys(columns + [stat, f'{stat}_scale']))
            common = frame.dropna(subset=required)
            if not np.isfinite(common[required].to_numpy(float)).all():
                raise ValueError(f'Nonfinite {stat} inputs')
            training = common[common.season.isin(TRAINING_SEASONS)]
            if set(training.season) != set(TRAINING_SEASONS):
                raise ValueError('All three training seasons must have eligible appearances')
            calibration_parts = []
            for earlier, later in calibration_folds(training):
                inner = SelectedBlendRegressor().fit(earlier[columns], earlier[stat])
                part = later[KEYS].copy()
                part['actual'], part['prediction'] = later[stat].to_numpy(), inner.predict(later[columns])
                part['line'], part['scale'] = later[f'{stat}_avg_season'].to_numpy(), later[f'{stat}_scale'].to_numpy()
                part['training_last_date'] = str(earlier.date.max().date())
                calibration_parts.append(part)
            calibration = pd.concat(calibration_parts, ignore_index=True)
            calibrator, quantiles, base_rate = calibration_fit(calibration)
            model = SelectedBlendRegressor().fit(training[columns], training[stat])
            artifact = {'artifact_version': 1, 'model': model, 'target': stat, 'features': columns,
                        'direction_calibrator': calibrator, 'interval_quantiles': quantiles,
                        'calibration_base_rate': base_rate, 'training_seasons': list(TRAINING_SEASONS),
                        'calibration_seasons': list(TRAINING_SEASONS[1:]),
                        'minimum_prior_games': min_prior_games, 'minimum_prior_avg_minutes': min_prior_avg_minutes,
                        'conditional_known_absences': True,
                        'direction_line': 'Pregame season-to-date player average, conditional on a non-push',
                        'uncertainty_scale': 'Prior ten-appearance stat standard deviation, floor one',
                        'calibration_method': 'Expanding chronological development predictions; transfer after refitting on all training seasons'}
            row = {'features': columns, 'train_rows': len(training), 'calibration_rows': len(calibration),
                   'training_dates': [str(training.date.min().date()), str(training.date.max().date())],
                   'evaluation': None}
            if evaluate:
                test = common[common.season.eq(TEST_SEASON)]
                if test.empty or training.date.max() >= test.date.min():
                    raise ValueError('Test appearances must follow training dates')
                metrics, prediction_frame = assess(test, model.predict(test[columns]), stat, calibrator, quantiles, base_rate)
                metrics['test_rows'] = len(test)
                metrics['dates'] = [str(test.date.min().date()), str(test.date.max().date())]
                row['evaluation'] = metrics
                predictions[stat] = prediction_frame
                print(f'{stat}: {len(training):,} train / {len(test):,} test; MAE={metrics["mae"]:.4f}; calibrated proxy accuracy={metrics["direction_confidence"]["accuracy"]:.2%}', flush=True)
            else:
                print(f'{stat}: trained {len(training):,} appearances, {len(columns)} features; calibration from {len(calibration):,} out-of-time predictions.', flush=True)
            results[stat] = {'artifact': artifact, 'report': row}
            calibration_outputs[stat] = calibration
    report = {'model': 'selected_blend', 'model_parameters': SelectedBlendRegressor().get_params(),
              'training_seasons': list(TRAINING_SEASONS), 'evaluation_season': TEST_SEASON if evaluate else None,
              'test_used': evaluate, 'database_writes': False, 'stats': {s: r['report'] for s,r in results.items()},
              'limitations': ['Absence features require supplied historical absences; they are not timestamped pregame injury reports.',
                              'Positions use current player metadata; stint boundary/transition gaps remain excluded.',
                              'Confidence concerns a season-average proxy, not actual betting lines.',
                              'Temporal calibration transfers after refitting; interval coverage is empirical, not guaranteed.',
                              '2025-26 has been evaluated previously; further validation needs future data.']}
    return results, report, predictions, calibration_outputs, snapshot
