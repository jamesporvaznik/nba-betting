"""Legacy nine-feature comparisons, retained as an explicit evaluation option."""

import pandas as pd
from models.evaluation.metrics import evaluate_over_under, evaluate_regression
from models.training.xgboost_model import train_xgboost
from models.training.baseline import train_baseline
from models.features.engineering import engineer_features, get_feature_cols, eligible_rows
from models.data.load import load_gamelogs, TRAINING_SEASONS, TEST_SEASON

STATS = ['points', 'rebounds', 'assists']


def run_baselines(stats=None, min_prior_games=20, min_prior_avg_minutes=10.0, snapshot=None):
    save = False  # Main entry point owns versioned output; never overwrite old model files.
    # Load data
    all_seasons = TRAINING_SEASONS + [TEST_SEASON]

    df = load_gamelogs(seasons=all_seasons) if snapshot is None else pd.DataFrame(snapshot['players'])
    df = df[df.season.isin(all_seasons)].copy()
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
    raw_count = len(df)
    df = eligible_rows(df, min_prior_games, min_prior_avg_minutes)
    print(f'Pregame-eligible rows: {len(df):,}/{raw_count:,}; at least {min_prior_games} '
          f'prior season appearances and {min_prior_avg_minutes:g} prior average minutes.')

    # Train/test split — chronological
    train_df = df[df['season'].isin(TRAINING_SEASONS)]
    test_df = df[df['season'] == TEST_SEASON]
    print(f"\nTrain rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    # Train a model for each stat
    results = {}
    for stat in stats or STATS:
        print(f"\n{'='*50}")
        print(f"STAT: {stat.upper()}")
        print('='*50)

        feature_cols = get_feature_cols(stat)

        # Drop rows missing features or target
        cols_needed = feature_cols + [stat]
        tr = train_df.dropna(subset=cols_needed)
        te = test_df.dropna(subset=cols_needed)
        if tr.empty or te.empty:
            raise ValueError(f'No usable training/test rows for {stat}')
        if tr['date'].max() >= te['date'].min():
            raise ValueError(
                'Training dates must precede the held-out test season')

        X_train = tr[feature_cols]
        y_train = tr[stat]
        X_test = te[feature_cols]
        y_test = te[stat]
        # Same player-specific, prior-only proxy line for every model.
        proxy_lines = te[f'{stat}_avg_season']
        rolling_preds = te[f'{stat}_avg5'].to_numpy()
        rolling_metrics = evaluate_regression(
            y_test, rolling_preds, label=f'Last-five average ({stat})')
        rolling_accuracy = evaluate_over_under(y_test, rolling_preds, lines=proxy_lines,
                                               label=f'Last-five average ({stat})')

        # Baseline (optional)
        baseline_model, baseline_preds = train_baseline(
            X_train, y_train, X_test, y_test, stat=stat
        )

        # XGBoost
        xgb_model, xgb_preds = train_xgboost(
            X_train, y_train, X_test, y_test, stat=stat, save=save
        )

        # Compare directions around the same pregame-average proxy, not betting lines.
        print(f"\n{'='*50}")
        print(f"PREGAME-AVERAGE PROXY COMPARISON FOR {stat.upper()}")
        print('='*50)

        # Baseline over/under
        baseline_accuracy = evaluate_over_under(
            y_test, baseline_preds, lines=proxy_lines, label=f"Baseline ({stat})")

        # XGBoost over/under
        xgb_accuracy = evaluate_over_under(
            y_test, xgb_preds, lines=proxy_lines, label=f"XGBoost ({stat})")

        results[stat] = {
            'baseline_model': baseline_model,
            'xgb_model':      xgb_model,
            'baseline_over_under': baseline_accuracy,
            'xgb_over_under': xgb_accuracy,
            'rolling_over_under': rolling_accuracy,
            'rolling_metrics': rolling_metrics,
            'baseline_metrics': evaluate_regression(y_test, baseline_preds, label=f'Linear summary ({stat})'),
            'xgb_metrics': evaluate_regression(y_test, xgb_preds, label=f'XGBoost summary ({stat})'),
            'train_rows': len(tr), 'test_rows': len(te), 'features': feature_cols,
        }

    print("\nTraining complete." +
          (" Models saved to models/artifacts/." if save else " No models overwritten."))

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - DIRECTION VERSUS PREGAME-AVERAGE PROXY (NOT SPORTSBOOK LINES)")
    print("="*60)
    for stat in results:
        print(
            f"{stat.upper():10} | Last-five: {results[stat]['rolling_over_under']:.1%} | Linear: {results[stat]['baseline_over_under']:.1%} | XGBoost: {results[stat]['xgb_over_under']:.1%}")

    return results
