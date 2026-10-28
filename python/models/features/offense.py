"""Prior-only shot volume and offensive involvement, from game box scores."""

import numpy as np
import pandas as pd

from models.features.engineering import chronological
from models.features.matchup import experimental_groups


def add_offense_features(frame):
    frame = chronological(frame)
    count_cols = ['fga', 'fta', '3PA', 'assists', 'turnovers']
    frame[count_cols] = frame[count_cols].apply(pd.to_numeric)
    # Complete team-game denominators before any eligibility filtering. Shares
    # describe each completed appearance, and are shifted before becoming X.
    team_keys = ['season', 'team', 'date', 'gameid']
    totals = frame.groupby(team_keys)[['fga', 'fta', 'assists']].transform('sum')
    for col in ['fga', 'fta', 'assists']:
        frame[f'raw_{col}_team_share'] = frame[col].div(totals[col].replace(0, np.nan)).where(totals[col] > 0, 0)
    groups = frame.groupby(['playerid', 'season'])
    for col in ['fga', 'fta', '3PA']:
        name = col.lower()
        for window in [5, 10, 'season']:
            frame[f'{name}_prior{window}'] = groups[col].transform(
                lambda x: x.shift(1).expanding().mean() if window == 'season'
                else x.shift(1).rolling(window, min_periods=1).mean())
        frame[f'{name}_trend'] = frame[f'{name}_prior5'] - frame[f'{name}_priorseason']
    minutes = groups['minutes'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    for col in ['fga', 'fta', '3PA', 'turnovers']:
        counts = groups[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
        frame[f'{col.lower()}_per_minute_prior10'] = counts / minutes.replace(0, np.nan)
    for col in ['fga', 'fta', 'assists']:
        for window in [5, 10, 'season']:
            frame[f'{col}_team_share_prior{window}'] = groups[f'raw_{col}_team_share'].transform(
                lambda x: x.shift(1).expanding().mean() if window == 'season'
                else x.shift(1).rolling(window, min_periods=1).mean())
        frame[f'{col}_team_share_trend'] = frame[f'{col}_team_share_prior5'] - frame[f'{col}_team_share_priorseason']
    return frame.drop(columns=[f'raw_{col}_team_share' for col in ['fga', 'fta', 'assists']])


def offense_groups(stat):
    opportunity = experimental_groups(stat)['opportunity']
    shots = ['fga_prior5', 'fga_prior10', 'fga_priorseason', 'fta_prior5',
             'fta_prior10', 'fta_priorseason', '3pa_prior10', 'fga_trend', 'fta_trend',
             'fga_per_minute_prior10', 'fta_per_minute_prior10', '3pa_per_minute_prior10']
    involvement = ['fga_team_share_prior10', 'fta_team_share_prior10', 'assists_team_share_prior10',
                   'fga_team_share_trend', 'fta_team_share_trend', 'assists_team_share_trend',
                   'turnovers_per_minute_prior10']
    return {'baseline': [], 'opportunity': opportunity, 'shot_volume': shots,
            'involvement': involvement, 'opportunity_shots': opportunity + shots,
            'combined': opportunity + shots + involvement}
