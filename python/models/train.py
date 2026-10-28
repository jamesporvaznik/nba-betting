"""
Main training script.
Run with: python3 -m models.train
"""
from models.evaluation.metrics import evaluate_over_under, calibration_report
from models.training.xgboost_model import train_xgboost
from models.training.baseline import train_baseline
from models.features.engineering import engineer_features, get_feature_cols
from models.data.load import load_gamelogs, TRAINING_SEASONS, TEST_SEASON
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Stats to train models for
STATS = ['points', 'rebounds', 'assists']


def run():
    # Load data
    all_seasons = TRAINING_SEASONS + [TEST_SEASON]

    df = load_gamelogs(
        seasons=all_seasons,
        min_games_per_season=20,
        min_minutes=10.0
    )
    # player_df = load_players()
    # team_df = load_teams()
    # clusters_df = load_clusters()
    # player_shots_df = load_player_shots()
    # team_shots_df = load_team_shots()
    # player_plays_df = load_player_plays()
    # team_plays_df = load_team_plays()
    # team_stats_df = load_team_defensive_stats()

    # Feature engineering
    df = engineer_features(df)

    # Train/test split — chronological
    train_df = df[df['season'].isin(TRAINING_SEASONS)]
    test_df = df[df['season'] == TEST_SEASON]
    print(f"\nTrain rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    # Train a model for each stat
    results = {}
    for stat in STATS:
        print(f"\n{'='*50}")
        print(f"STAT: {stat.upper()}")
        print('='*50)

        feature_cols = get_feature_cols(stat)

        # Drop rows missing features or target
        cols_needed = feature_cols + [stat]
        tr = train_df.dropna(subset=cols_needed)
        te = test_df.dropna(subset=cols_needed)

        X_train = tr[feature_cols]
        y_train = tr[stat]
        X_test = te[feature_cols]
        y_test = te[stat]

        # Baseline (optional)
        baseline_model, baseline_preds = train_baseline(
            X_train, y_train, X_test, y_test, stat=stat
        )

        # XGBoost
        xgb_model, xgb_preds = train_xgboost(
            X_train, y_train, X_test, y_test, stat=stat, save=True
        )

        # Additional: Compare over/under accuracy between baseline and XGBoost
        print(f"\n{'='*50}")
        print(f"OVER/UNDER COMPARISON FOR {stat.upper()}")
        print('='*50)

        # Baseline over/under
        baseline_threshold = y_train.mean()
        baseline_accuracy = evaluate_over_under(
            y_test, baseline_preds, label=f"Baseline ({stat})")

        # XGBoost over/under
        xgb_accuracy = evaluate_over_under(
            y_test, xgb_preds, label=f"XGBoost ({stat})")

        # Calibration for XGBoost (binary classification on over/under)
        y_test_binary = (y_test > baseline_threshold).astype(int)
        xgb_proba = 1 / (1 + np.exp(-(xgb_preds - baseline_threshold)))

        print(f"\n{'='*50}")
        print(f"CALIBRATION FOR {stat.upper()}")
        print('='*50)
        calibration_report(y_test_binary, xgb_proba, n_buckets=5)

        results[stat] = {
            'baseline_model': baseline_model,
            'xgb_model':      xgb_model,
            'baseline_over_under': baseline_accuracy,
            'xgb_over_under': xgb_accuracy,
        }

    print("\n✅ Training complete. Models saved to models/artifacts/")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - OVER/UNDER ACCURACY")
    print("="*60)
    for stat in STATS:
        print(
            f"{stat.upper():10} | Baseline: {results[stat]['baseline_over_under']:.1%} | XGBoost: {results[stat]['xgb_over_under']:.1%}")

    return results


if __name__ == '__main__':
    # Need to import numpy for the calibration sigmoid
    import numpy as np
    run()
