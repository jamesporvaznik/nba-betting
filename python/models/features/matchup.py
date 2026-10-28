"""Experimental pregame pace, defense, and player opportunity features."""

import numpy as np
import pandas as pd

from models.features.engineering import chronological


def add_matchup_features(players, teams):
    """Join strictly earlier team history; current team outcomes never enter X."""
    teams = teams.copy()
    teams['date'] = pd.to_datetime(teams['date']).dt.normalize()
    if teams.duplicated(['season', 'team', 'date']).any():
        raise ValueError('Duplicate team/date histories require review')
    teams = teams.sort_values(['season', 'team', 'date'])
    numeric = ['team_pace', 'def_rating', 'opponent_fgm', 'opponent_fga',
               'opponent_3pm', 'opponent_3pa', 'opponent_fta',
               'opponent_pts', 'opponent_trb', 'opponent_ast']
    teams[numeric] = teams[numeric].apply(pd.to_numeric)
    groups = teams.groupby(['season', 'team'])
    history = teams[['season', 'team', 'date']].copy()
    for col in ['team_pace', 'def_rating', 'opponent_pts', 'opponent_trb', 'opponent_ast']:
        history[f'{col}_prior_season'] = groups[col].transform(lambda x: x.shift(1).expanding().mean())
        history[f'{col}_prior10'] = groups[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    totals = {col: groups[col].transform(lambda x: x.shift(1).expanding().sum())
              for col in ['opponent_fgm', 'opponent_fga', 'opponent_3pm', 'opponent_3pa', 'opponent_fta']}
    history['fg_pct_allowed_prior'] = totals['opponent_fgm'] / totals['opponent_fga'].replace(0, np.nan)
    history['three_pct_allowed_prior'] = totals['opponent_3pm'] / totals['opponent_3pa'].replace(0, np.nan)
    history['fta_per_fga_allowed_prior'] = totals['opponent_fta'] / totals['opponent_fga'].replace(0, np.nan)
    players = chronological(players)
    own = history[['season', 'team', 'date', 'team_pace_prior_season', 'team_pace_prior10']].rename(
        columns={'team_pace_prior_season': 'own_pace_prior_season', 'team_pace_prior10': 'own_pace_prior10'})
    result = players.merge(own, on=['season', 'team', 'date'], how='left', validate='many_to_one')
    opposing = history.rename(columns={'team': 'opponent', **{c: f'opp_{c}' for c in history.columns
                                                            if c not in ['season', 'team', 'date']}})
    result = result.merge(opposing, on=['season', 'opponent', 'date'], how='left', validate='many_to_one')
    result['expected_pace_proxy'] = (result['own_pace_prior_season'] + result['opp_team_pace_prior_season']) / 2
    result = chronological(result)
    groups = result.groupby(['playerid', 'season'])
    result['minutes_std10'] = groups['minutes'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())
    result['minutes_trend'] = result['minutes_avg5'] - result['minutes_avg_season']
    minute_totals = {w: groups['minutes'].transform(
        lambda x: x.shift(1).expanding().sum() if w == 'season' else x.shift(1).rolling(10, min_periods=1).sum())
        for w in ['season', '10']}
    for stat in ['points', 'rebounds', 'assists']:
        for window in ['season', '10']:
            totals = groups[stat].transform(lambda x: x.shift(1).expanding().sum() if window == 'season'
                                           else x.shift(1).rolling(10, min_periods=1).sum())
            result[f'{stat}_per_minute_{window}'] = totals / minute_totals[window].replace(0, np.nan)
        result[f'{stat}_opportunity_proxy'] = result[f'{stat}_per_minute_10'] * result['minutes_avg5']
        result[f'{stat}_pace_opportunity_proxy'] = result[f'{stat}_opportunity_proxy'] * result['expected_pace_proxy'] / 100
    return result.reset_index(drop=True)


def experimental_groups(stat):
    target = {'points': 'pts', 'rebounds': 'trb', 'assists': 'ast'}[stat]
    pace = ['own_pace_prior_season', 'own_pace_prior10', 'opp_team_pace_prior_season',
            'opp_team_pace_prior10', 'expected_pace_proxy']
    defense = ['opp_def_rating_prior_season', 'opp_def_rating_prior10',
               'opp_fg_pct_allowed_prior', 'opp_three_pct_allowed_prior',
               'opp_fta_per_fga_allowed_prior', f'opp_opponent_{target}_prior10']
    role = ['minutes_avg_season', 'minutes_std10', 'minutes_trend',
            f'{stat}_per_minute_season', f'{stat}_per_minute_10', f'{stat}_opportunity_proxy']
    return {'baseline': [], 'pace': pace, 'defense': defense, 'opportunity': role,
            'combined': pace + defense + role + [f'{stat}_pace_opportunity_proxy']}
