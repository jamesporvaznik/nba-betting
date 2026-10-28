"""Main training entry point; selected blend by default, explicit test evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from models.data.load import TRAINING_SEASONS, TEST_SEASON
from models.data.snapshot import capture_training_snapshot
from models.features.selected import STATS
from models.training.selected import train_selected


def run(stats=None, min_prior_games=20, min_prior_avg_minutes=10., save=True,
        model='blend', evaluate=False, snapshot=None, output_dir=None):
    if model not in ['blend', 'baseline']:
        raise ValueError('Model must be blend or baseline')
    if model == 'baseline' and not evaluate:
        raise ValueError('Legacy baseline comparisons require --evaluate')
    if not save and output_dir is not None:
        raise ValueError('--output-dir cannot be used with --no-save')
    stats = STATS if stats is None else list(stats)
    if not stats or set(stats)-set(STATS) or len(stats) != len(set(stats)):
        raise ValueError('Choose unique targets from points, rebounds, assists')
    if min_prior_games < 0 or min_prior_avg_minutes < 0:
        raise ValueError('Eligibility thresholds must be nonnegative')
    folder = None
    if save:
        folder = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent/'artifacts'/f'train_{model}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
        if folder.exists():
            raise FileExistsError(f'Output directory already exists: {folder}')
    if model == 'baseline':
        from models.training.legacy import run_baselines
        results = run_baselines(stats, min_prior_games, min_prior_avg_minutes, snapshot=snapshot)
        if save:
            folder.mkdir(parents=True, exist_ok=False)
            report = {'model': 'legacy_baselines', 'evaluation_season': TEST_SEASON, 'stats': {}}
            for stat, row in results.items():
                for name in ['baseline', 'xgb']:
                    joblib.dump(row[f'{name}_model'], folder/f'{name}_{stat}.joblib')
                report['stats'][stat] = {key: row[key] for key in ['features','train_rows','test_rows','baseline_metrics','xgb_metrics']}
            (folder/'report.json').write_text(json.dumps(report, indent=2))
            print(f'Baseline comparison saved to {folder}')
        return results
    if snapshot is None:
        seasons = TRAINING_SEASONS + ([TEST_SEASON] if evaluate else [])
        snapshot = capture_training_snapshot(seasons)
    results, report, predictions, calibration, used_snapshot = train_selected(
        snapshot, stats, min_prior_games, min_prior_avg_minutes, evaluate)
    if save:
        folder.mkdir(parents=True, exist_ok=False)
        source = folder/'source_snapshot.json'
        source.write_text(json.dumps(used_snapshot, default=str))
        report['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        for stat, row in results.items():
            joblib.dump(row['artifact'], folder/f'blend_{stat}.joblib')
            calibration[stat].to_csv(folder/f'calibration_{stat}.csv', index=False)
            if evaluate:
                predictions[stat].to_csv(folder/f'predictions_{stat}.csv', index=False)
        (folder/'report.json').write_text(json.dumps(report, indent=2))
        (folder/'features.json').write_text(json.dumps({s:r['report']['features'] for s,r in results.items()}, indent=2))
        if evaluate:
            pd.DataFrame([{'stat': stat, 'mae': row['report']['evaluation']['mae'],
                           'rmse': row['report']['evaluation']['rmse'],
                           'bias': row['report']['evaluation']['bias'],
                           'accuracy': row['report']['evaluation']['direction_confidence']['accuracy'],
                           'brier': row['report']['evaluation']['direction_confidence']['brier_score'],
                           'coverage_90': row['report']['evaluation']['intervals'][-1]['observed_coverage']}
                          for stat,row in results.items()]).to_csv(folder/'summary.csv',index=False)
            pd.DataFrame([{'stat': stat, **bucket} for stat,row in results.items()
                          for bucket in row['report']['evaluation']['direction_confidence']['buckets']]).to_csv(folder/'confidence_buckets.csv',index=False)
        print(f'Training artifacts saved to {folder}')
    else:
        print('Training complete; --no-save wrote no files.')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['blend','baseline'], default='blend')
    parser.add_argument('--stats', nargs='+', choices=STATS, default=STATS)
    parser.add_argument('--evaluate', action='store_true', help='Explicitly score the fixed 2025-26 test season')
    parser.add_argument('--snapshot', type=Path, help='Use a saved database snapshot offline')
    parser.add_argument('--output-dir', type=Path, help='New output directory; default creates a dated run')
    parser.add_argument('--min-prior-games', type=int, default=20)
    parser.add_argument('--min-prior-avg-minutes', type=float, default=10.)
    parser.add_argument('--no-save', action='store_true', help='Fit in memory without writing files')
    args = parser.parse_args()
    if args.model == 'baseline' and not args.evaluate:
        parser.error('--model baseline requires --evaluate')
    if args.no_save and args.output_dir:
        parser.error('--output-dir cannot be used with --no-save')
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else None
    run(args.stats, args.min_prior_games, args.min_prior_avg_minutes, not args.no_save,
        args.model, args.evaluate, snapshot, args.output_dir)


if __name__ == '__main__':
    main()
