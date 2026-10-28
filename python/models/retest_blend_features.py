"""Fixed-blend feature retests on development seasons, including timing diagnostics."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from models.compare_algorithms import blend_prediction, calibrating_rows
from models.experiment_features import paired_date_interval
from models.experiment_playtypes import chronological_folds
from models.features.matchup import experimental_groups
from models.features.playtype_matchup import add_playtype_matchups, playtype_columns
from models.features.shot_zone_matchup import add_shot_zone_matchups, shot_zone_columns
from models.full_training_review import TRAIN, STATS, KEYS, prepare_features, development_folds, calibration_fit, assess
from nba.repair_shot_locations import connection


def development_only(snapshot):
    output = dict(snapshot)
    for key in ['players', 'teams', 'absences']:
        output[key] = [r for r in snapshot[key] if r['season'] in TRAIN]
    return output


def capture_profiles():
    queries = {
        'playtype_players': 'SELECT player_id,season,play_type,poss_pct FROM public.player_playtype_stats WHERE season=ANY(%s) ORDER BY season,player_id,play_type',
        'playtype_defense': 'SELECT team_abbreviation AS opponent,season,play_type,points_allowed_per_possession AS ppp,possessions_defended FROM public.team_defensive_playtype_stats WHERE season=ANY(%s) ORDER BY season,team_abbreviation,play_type',
        'zone_players': 'SELECT player_id,season,shot_zone,frequency_pct FROM public.player_shot_locations WHERE season=ANY(%s) ORDER BY season,player_id,shot_zone',
        'zone_defense': 'SELECT t.abbreviation AS opponent,s.season,s.shot_zone,s.fg_pct_allowed,s.frequency_pct FROM public.team_defensive_shot_locations s JOIN public.teams t ON t.nba_api_id=s.team_id WHERE s.season=ANY(%s) ORDER BY s.season,t.abbreviation,s.shot_zone',
    }
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for key, query in queries.items():
                cursor.execute(query, (TRAIN,))
                snapshot[key] = [dict(row) for row in cursor.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def inner_calibration_split(training):
    dates = np.sort(training.date.unique())
    if len(dates) < 8:
        raise ValueError('Insufficient training dates for temporal calibration')
    cutoff = dates[int(len(dates)*.75)]
    earlier, calibration = training[training.date < cutoff], training[training.date >= cutoff]
    if earlier.empty or calibration.empty or earlier.date.max() >= calibration.date.min():
        raise ValueError('Invalid inner calibration chronology')
    return earlier, calibration


def choose_addition(rows, stat):
    """Require improvements in both folds and >=.1% mean relative MAE reduction."""
    frame = pd.DataFrame([r for r in rows if r['stat'] == stat])
    reference = frame[frame.variant.eq('baseline')].set_index('fold').mae
    choices = [(0., 'baseline')]
    for name, group in frame.groupby('variant', sort=False):
        if name == 'baseline':
            continue
        if set(group.fold) != set(reference.index) or len(group) != len(reference):
            raise ValueError('Mismatched selection folds')
        relative = group.set_index('fold').mae / reference - 1
        if (relative < 0).all() and relative.mean() <= -.001:
            choices.append((float(relative.mean()), name))
    return min(choices)[1]


def fit_blend(train, validation, inputs, stat, config):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x_train = imputer.fit_transform(train[inputs])
    x_validation = imputer.transform(validation[inputs])
    linear = LinearRegression().fit(x_train, train[stat])
    tree = CatBoostRegressor(**config['params'], random_seed=42, thread_count=1,
                             verbose=False, allow_writing_files=False).fit(x_train, train[stat])
    return blend_prediction(linear.predict(x_validation), tree.predict(x_validation), config['tree_weight'])


def baseline_signature(train, validation, inputs, stat):
    digest = hashlib.sha256()
    for frame in [train, validation]:
        cols = list(dict.fromkeys(KEYS + inputs + [stat, f'{stat}_avg_season', f'{stat}_scale']))
        digest.update(pd.util.hash_pandas_object(frame[cols], index=False).values.tobytes())
    return digest.hexdigest()


def compare_groups(frame, base, groups, folds, stat, config, kind, timing, folder, baseline_cache, flags=None):
    rows, coverage, predictions = [], [], {}
    for fold, train, validation in folds:
        if train.empty or validation.empty or train.date.max() >= validation.date.min():
            raise ValueError('Invalid outer chronology')
        earlier, calibration = inner_calibration_split(train)
        item = {'kind': kind, 'timing': timing, 'stat': stat, 'fold': fold,
                'train_rows': len(train), 'validation_rows': len(validation),
                'inner_train_rows': len(earlier), 'calibration_rows': len(calibration),
                'train_last_date': str(train.date.max().date()), 'validation_first_date': str(validation.date.min().date())}
        if flags:
            item.update({f'validation_missing_{flag}': int(validation[flag].eq(0).sum()) for flag in flags})
        coverage.append(item)
        references, reference_probabilities = {}, {}
        for variant, extras in groups.items():
            inputs = list(dict.fromkeys(base + extras))
            if not np.isfinite(train[base].to_numpy(float)).all() or not np.isfinite(validation[base].to_numpy(float)).all():
                raise ValueError('Invalid baseline inputs')
            signature = baseline_signature(train, validation, inputs, stat) if variant == 'baseline' else None
            if signature is not None and signature in baseline_cache:
                prediction, classifier, quantiles, cal_rate = baseline_cache[signature]
            else:
                cal_prediction = fit_blend(earlier, calibration, inputs, stat, config)
                cal_frame = calibrating_rows(calibration, cal_prediction, stat, earlier.date.max())
                classifier, quantiles, cal_rate = calibration_fit(cal_frame)
                prediction = fit_blend(train, validation, inputs, stat, config)
                if signature is not None:
                    baseline_cache[signature] = prediction, classifier, quantiles, cal_rate
            row, output = assess(validation, prediction, stat, classifier, quantiles, cal_rate)
            errors = abs(validation[stat].to_numpy()-prediction)
            references[variant] = errors
            reference_probabilities[variant] = output.probability_over_proxy_nonpush.to_numpy()
            row.update({'kind': kind, 'timing': timing, 'retrospective_lookahead': timing == 'completed_current',
                        'stat': stat, 'fold': fold, 'variant': variant, 'features': inputs,
                        'train_rows': len(train), 'validation_rows': len(validation),
                        'mae_change_vs_baseline': float(np.mean(errors-references['baseline'])),
                        'paired_95pct_vs_baseline': paired_date_interval(validation.date, references['baseline'], errors)})
            control = 'both_profiles' if variant == 'interactions' else 'availability_only' if variant in ['player_profile','opponent_profile','both_profiles'] else None
            if control:
                row.update({'incremental_control': control,
                            'incremental_mae_change': float(np.mean(errors-references[control])),
                            'incremental_paired_95pct': paired_date_interval(validation.date, references[control], errors)})
            for flag in flags or []:
                output[flag] = validation[flag].to_numpy()
            output.to_csv(folder/f'predictions_{kind}_{timing}_{stat}_{fold}_{variant}.csv', index=False)
            predictions[(fold, variant)] = (validation[KEYS].reset_index(drop=True), errors)
            rows.append(row)
            print(f'{kind} {timing} {stat} {fold} {variant}: MAE={row["mae"]:.4f}; delta={row["mae_change_vs_baseline"]:+.4f}; calibrated proxy={row["direction_confidence"]["accuracy"]:.2%}',flush=True)
    return rows, coverage, predictions


def run(source, blend_source, profiles, folder):
    profile_keys = ['playtype_players','playtype_defense','zone_players','zone_defense']
    if any({r['season'] for r in profiles[key]} - set(TRAIN) for key in profile_keys):
        raise ValueError('Profile seasons must be within development seasons')
    source_file = source/'source_snapshot.json'
    manifest = json.loads((source/'frozen_features.json').read_text())
    blend_manifest = json.loads((blend_source/'frozen_comparison.json').read_text())
    grid = {c['id']: c for c in json.loads((blend_source/'design.json').read_text())['candidate_grid']}
    # Filter all outcome-bearing source tables before engineering any history.
    snapshot = development_only(json.loads(source_file.read_text()))
    frame = prepare_features(snapshot, expected_seasons=TRAIN)
    if set(frame.season) - set(TRAIN):
        raise ValueError('Unexpected engineered season')
    configs = {stat: {'params': grid[blend_manifest['blends'][stat]['tree_id']]['params'],
                      'tree_weight': blend_manifest['blends'][stat]['tree_weight']} for stat in STATS}
    results, coverage, chosen_pace, timing_comparison, baseline_cache = [], [], {}, [], {}
    with threadpool_limits(limits=1):
        for stat in STATS:
            base = manifest['selections'][stat]['features']
            prior = experimental_groups(stat)
            groups = {'baseline': [], 'pace': prior['pace'], 'defense': prior['defense'],
                      'pace_and_defense': prior['pace'] + prior['defense']}
            common = frame.dropna(subset=list(dict.fromkeys(base + groups['pace_and_defense'] + [stat, f'{stat}_scale'])))
            folds = [(f'full_{season}', tr, va) for season,tr,va in development_folds(common)]
            rows, counts, _ = compare_groups(common, base, groups, folds, stat, configs[stat],
                                             'pace_defense', 'prior_games', folder, baseline_cache)
            results.extend(rows); coverage.extend(counts)
            choice = choose_addition(rows, stat)
            chosen_pace[stat] = {'variant': choice, 'features': groups[choice]}
            print(f'Pace/defense selection for {stat}: {choice}',flush=True)
        (folder/'pace_selection.json').write_text(json.dumps(chosen_pace,indent=2))
        # Profile experiments target points, using the previously selected pace
        # addition identically in every profile timing/variant comparison.
        stat = 'points'
        base = list(dict.fromkeys(manifest['selections'][stat]['features'] + chosen_pace[stat]['features']))
        common = frame.dropna(subset=base + [stat, f'{stat}_scale'])
        for kind in ['playtype','shot_zone']:
            timings = {}
            for timing in ['previous','completed_current']:
                if kind == 'playtype':
                    enriched = add_playtype_matchups(common, pd.DataFrame(profiles['playtype_players']),
                                                     pd.DataFrame(profiles['playtype_defense']), timing)
                    groups = playtype_columns(timing)
                    prefix = 'prior' if timing == 'previous' else 'retrospective'
                    flags = [f'{prefix}_player_profile_available', f'{prefix}_opponent_profile_available']
                else:
                    enriched = add_shot_zone_matchups(common, pd.DataFrame(profiles['zone_players']),
                                                      pd.DataFrame(profiles['zone_defense']), timing)
                    groups = shot_zone_columns(timing)
                    prefix = 'lag_zone' if timing == 'previous' else 'retrospective_zone'
                    flags = [f'{prefix}_player_available', f'{prefix}_defense_available']
                groups = {'baseline': [], 'availability_only': flags,
                          **{k:v for k,v in groups.items() if k != 'baseline'}}
                rows, counts, predictions = compare_groups(enriched, base, groups, chronological_folds(enriched),
                                                            stat, configs[stat], kind, timing, folder, baseline_cache, flags)
                results.extend(rows); coverage.extend(counts)
                timings[timing] = predictions
            for key, (keys, errors) in timings['previous'].items():
                current_keys, current_errors = timings['completed_current'][key]
                if not keys.equals(current_keys):
                    raise ValueError('Profile timing cohorts differ')
                if key[1] == 'baseline' and not np.allclose(errors,current_errors,rtol=1e-10,atol=1e-10):
                    raise ValueError('Profile timing baseline predictions differ')
                timing_comparison.append({'kind':kind,'fold':key[0],'variant':key[1],
                                           'previous_mae':float(errors.mean()),'current_mae':float(current_errors.mean()),
                                           'delta_current_minus_previous':float(np.mean(current_errors-errors)),
                                           'paired_95pct_current_vs_previous':paired_date_interval(keys.date,errors,current_errors)})
    report = {'database_writes':False,'production_models_overwritten':False,
              'held_out_2025_26_engineered_or_scored':False,
              'model_configs':configs,'pace_selection':chosen_pace,'coverage':coverage,
              'profile_timing_comparison':timing_comparison,
              'confidence_method':'Each outer training set is split by date: first 75% of unique dates fit an inner blend, last 25% calibrate direction and intervals. Refit the same blend on the entire outer training set before validation. No outer-validation outcomes calibrate its probabilities.',
              'limitations':['Development-only exploratory comparisons: model/features were previously selected using these seasons, so gains require later confirmation.',
                             'Completed-current profiles include scored and future games. Their results are retrospective diagnostics and do not enter production feature selection.',
                             'Prior profiles lag one season. No 2021-22 profiles exist; matched profile splits use early 2023-24 -> late 2023-24 and 2023-24 -> 2024-25.',
                             'Pace/defense uses full expanding folds. Raw profile scores are not comparable with full-training pace scores; each has a matched baseline.',
                             'Profiles are missing for some players; all prediction rows remain, using training-only median imputation and availability flags.',
                             'Known historical absences and current master positions remain conditional inputs; profile source coverage is not necessarily every game.',
                             'Confidence targets season-average proxy direction conditional on non-pushes, not sportsbook lines. Temporal calibration transfers after refitting, without guaranteed interval coverage.',
                             'Paired date-block bootstrap intervals are exploratory without multiple-comparison or selection adjustment.',
                             'The reused source JSON contains four seasons; all game/team/absence records are filtered to the first three before engineering. No 2025-26 profile query or scoring occurs.'],
              'results':results}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame([{'kind':r['kind'],'timing':r['timing'],'retrospective':r['retrospective_lookahead'],
                   'stat':r['stat'],'fold':r['fold'],'variant':r['variant'],
                   'mae':r['mae'],'rmse':r['rmse'],'bias':r['bias'],
                   'accuracy':r['direction_confidence']['accuracy'],'brier':r['direction_confidence']['brier_score'],
                   'log_loss':r['direction_confidence']['log_loss'],'coverage_90':r['intervals'][-1]['observed_coverage'],
                   'mae_change_vs_baseline':r['mae_change_vs_baseline'],
                   'paired_95pct_vs_baseline':r['paired_95pct_vs_baseline'],
                   'incremental_control':r.get('incremental_control'),
                   'incremental_mae_change':r.get('incremental_mae_change')} for r in results]).to_csv(folder/'results.csv',index=False)
    pd.DataFrame(timing_comparison).to_csv(folder/'timing_comparison.csv',index=False)
    pd.DataFrame([{'kind':r['kind'],'timing':r['timing'],'stat':r['stat'],'fold':r['fold'],'variant':r['variant'],**b}
                  for r in results for b in r['direction_confidence']['buckets']]).to_csv(folder/'confidence_buckets.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-source',required=True,type=Path)
    parser.add_argument('--blend-source',required=True,type=Path)
    parser.add_argument('--profile-snapshot',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    profiles=json.loads(args.profile_snapshot.read_text()) if args.profile_snapshot else capture_profiles()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    path=args.output_dir/'profile_snapshot.json'
    path.write_text(json.dumps(profiles,indent=2))
    run(args.training_source,args.blend_source,profiles,args.output_dir)
    report=args.output_dir/'report.json'
    data=json.loads(report.read_text())
    data['source']={'profile_source':'live_database' if not args.profile_snapshot else str(args.profile_snapshot),
                    'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'game_snapshot':str(args.training_source/'source_snapshot.json'),
                    'game_sha256':hashlib.sha256((args.training_source/'source_snapshot.json').read_bytes()).hexdigest(),
                    'blend_manifest_sha256':hashlib.sha256((args.blend_source/'frozen_comparison.json').read_bytes()).hexdigest()}
    report.write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
