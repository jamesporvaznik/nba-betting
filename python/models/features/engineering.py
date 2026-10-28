import pandas as pd

# Stats we want to build rolling features for
ROLLING_STATS = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes', 'turnovers']
WINDOWS       = [5, 10]


def chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Order appearances by date; ambiguous same-day player records need review."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    if df[['playerid', 'season', 'date']].isna().any().any():
        raise ValueError('Missing player, season, or date')
    if df.duplicated(['playerid', 'season', 'date']).any():
        raise ValueError('Duplicate player appearances on the same date')
    return df.sort_values(['playerid', 'date'], kind='stable')

# get rolling averages
def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average features per player. Uses shift(1) to prevent data leakage."""
    df = chronological(df)
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        grp = df.groupby(['playerid', 'season'])[stat]
        for w in WINDOWS:
            df[f'{stat}_avg{w}'] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        # Season average up to but not including current game
        df[f'{stat}_avg_season'] = grp.transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
    groups = df.groupby(['playerid', 'season'])
    df['prior_games_season'] = groups.cumcount()
    return df

# add days of rest as a feature
def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days of rest and back-to-back flag."""
    df = chronological(df)
    df['days_rest'] = (
        df.groupby(['playerid', 'season'])['date']
        .diff()
        .dt.days - 1
    ).clip(0, 14)  # cap at 14 days
    df['is_back_to_back'] = (df['days_rest'] == 0).astype(int)
    return df

# add home vs away as a feature
def add_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add home/away and result encoding."""
    df = df.copy()
    df['is_home'] = (df['homeaway'].str.strip() == 'H').astype(int)
    return df

def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prior team-game totals allowed, never an average over individual players.
    Aggregate complete games first, then average strictly earlier dates within
    each opponent/season. All current-day outcomes are excluded together.
    """
    df = chronological(df)
    df['opponent'] = df['opponent'].str.strip()
    stats = [s for s in ['points', 'rebounds', 'assists'] if s in df.columns]
    game_key = 'gameid' if 'gameid' in df.columns else 'team'
    if game_key not in df.columns:
        raise ValueError('Opponent histories require gameid or team to aggregate games')
    keys = ['opponent', 'season', 'date']
    games = df.groupby(keys + [game_key], as_index=False)[stats].sum(min_count=1)
    # Multiple games on a date, if present, share the same prior-only baseline.
    daily_sum = games.groupby(keys)[stats].sum(min_count=1)
    daily_count = games.groupby(keys)[stats].count()
    prior_sum = daily_sum.groupby(level=['opponent', 'season']).transform(
        lambda x: x.cumsum().shift(1))
    prior_count = daily_count.groupby(level=['opponent', 'season']).transform(
        lambda x: x.cumsum().shift(1))
    baselines = (prior_sum / prior_count).rename(
        columns={s: f'opp_{s}_allowed_avg' for s in stats}).reset_index()
    return df.merge(baselines, on=keys, how='left', validate='many_to_one').sort_values(
        ['playerid', 'date'], kind='stable').reset_index(drop=True)


def eligible_rows(df: pd.DataFrame, min_prior_games: int = 20,
                  min_prior_avg_minutes: float = 10.0) -> pd.DataFrame:
    """Select prediction rows using only already-engineered pregame history."""
    if min_prior_games < 0 or min_prior_avg_minutes < 0:
        raise ValueError('Eligibility thresholds must be nonnegative')
    mask = df['prior_games_season'] >= min_prior_games
    if min_prior_avg_minutes > 0:
        mask &= df['minutes_avg_season'] >= min_prior_avg_minutes
    return df.loc[mask].copy()

# run all the feature creation functions
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in order."""
    print("Engineering features...")
    df = add_rolling_averages(df)
    df = add_rest_features(df)
    df = add_game_context(df)
    df = add_opponent_features(df)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df

# get feature columns
def get_feature_cols(stat: str = 'points') -> list[str]:
    """Return the list of feature columns to use for a given stat."""
    base = [
        f'{stat}_avg5',
        f'{stat}_avg10',
        f'{stat}_avg_season',
        'minutes_avg5',
        'minutes_avg10',
        'days_rest',
        'is_back_to_back',
        'is_home',
        f'opp_{stat}_allowed_avg',
    ]
    return [col for col in base if col in get_all_possible_cols()]

# get all possible features
def get_all_possible_cols() -> list[str]:
    """All columns that could be generated by engineer_features."""
    cols = []
    for stat in ROLLING_STATS:
        for w in WINDOWS:
            cols.append(f'{stat}_avg{w}')
        cols.append(f'{stat}_avg_season')
    cols += ['minutes_avg5', 'minutes_avg10', 'days_rest', 'is_back_to_back', 'is_home']
    for stat in ['points', 'rebounds', 'assists']:
        cols.append(f'opp_{stat}_allowed_avg')
    return cols
