"""Prior-game team rebounding context and lagged player tracking profiles."""

import numpy as np
import pandas as pd

from models.features.engineering import chronological
from models.features.playtype_matchup import next_season

TRACKING = ['reb_chance_pct', 'dreb_chance_pct', 'oreb_chance_pct', 'reb_contest_pct']
COUNTS = ['team_orb', 'team_dreb', 'opponent_orb', 'opponent_dreb', 'team_fg_misses', 'opponent_fg_misses']
RATES = ['offensive_reb_rate', 'defensive_reb_rate']


def rebound_columns():
    team = [f'opp_reb_{c}_prior{w}' for c in COUNTS + RATES for w in ['season', '10']]
    team += [f'own_reb_{c}_prior10' for c in ['team_orb', 'team_dreb'] + RATES]
    tracking = ['lag_reb_tracking_available'] + [f'lag_track_{c}' for c in TRACKING]
    interactions = ['reb_offensive_matchup_interaction', 'reb_defensive_matchup_interaction']
    return {'baseline': [], 'team_rebounding': team, 'player_tracking': tracking,
            'both': team + tracking, 'interactions': team + tracking + interactions}


def add_team_rebounding_features(games, team_games):
    teams = team_games.copy()
    teams['date'] = pd.to_datetime(teams.date).dt.normalize()
    if teams.duplicated(['season', 'team', 'date']).any():
        raise ValueError('Duplicate team/date in rebounding history')
    columns = ['team_orb', 'team_trb', 'opponent_orb', 'opponent_trb', 'opponent_fga', 'opponent_fgm']
    teams[columns] = teams[columns].apply(pd.to_numeric)
    if not np.isfinite(teams[columns].to_numpy(dtype=float)).all() or (teams[columns] < 0).any().any():
        raise ValueError('Invalid rebounding team counts')
    # The reciprocal row supplies the team's own shooting, absent from this
    # table's own-row shooting columns. No current values are joined into X.
    mirror = teams[['season', 'date', 'team', 'opponent_abbrev', 'opponent_fga', 'opponent_fgm']].rename(
        columns={'team': 'opponent_abbrev', 'opponent_abbrev': 'team',
                 'opponent_fga': 'team_fga', 'opponent_fgm': 'team_fgm'})
    teams = teams.merge(mirror, on=['season', 'date', 'team', 'opponent_abbrev'],
                        how='left', validate='one_to_one')
    if teams[['team_fga', 'team_fgm']].isna().any().any():
        raise ValueError('Missing reciprocal team-game shooting row')
    teams['team_dreb'] = teams.team_trb-teams.team_orb
    teams['opponent_dreb'] = teams.opponent_trb-teams.opponent_orb
    teams['team_fg_misses'] = teams.team_fga-teams.team_fgm
    teams['opponent_fg_misses'] = teams.opponent_fga-teams.opponent_fgm
    if (teams[COUNTS] < 0).any().any():
        raise ValueError('Invalid derived rebound counts')
    teams = teams.sort_values(['season', 'team', 'date'])
    groups = teams.groupby(['season', 'team'])
    history = teams[['season', 'team', 'date']].copy()
    sums = {}
    for c in COUNTS:
        for w in ['season', '10']:
            history[f'{c}_prior{w}'] = groups[c].transform(
                lambda x: x.shift(1).expanding().mean() if w == 'season'
                else x.shift(1).rolling(10, min_periods=1).mean())
            if c in ['team_orb', 'team_dreb', 'opponent_orb', 'opponent_dreb']:
                sums[c, w] = groups[c].transform(lambda x: x.shift(1).expanding().sum() if w == 'season'
                                                else x.shift(1).rolling(10, min_periods=1).sum())
    for w in ['season', '10']:
        history[f'offensive_reb_rate_prior{w}'] = sums['team_orb', w] / (sums['team_orb', w]+sums['opponent_dreb', w]).replace(0, np.nan)
        history[f'defensive_reb_rate_prior{w}'] = sums['team_dreb', w] / (sums['team_dreb', w]+sums['opponent_orb', w]).replace(0, np.nan)
    opponent_history = history.rename(columns={'team': 'opponent', **{c: f'opp_reb_{c}' for c in history if c not in ['season', 'team', 'date']}})
    result = chronological(games).merge(opponent_history, on=['season', 'opponent', 'date'],
                                         how='left', validate='many_to_one')
    own_cols = [f'{c}_prior10' for c in ['team_orb', 'team_dreb'] + RATES]
    own_history = history[['season', 'team', 'date'] + own_cols].rename(columns={c: f'own_reb_{c}' for c in own_cols})
    result = result.merge(own_history, on=['season', 'team', 'date'], how='left', validate='many_to_one')
    return result


def add_rebounding_features(games, team_games, tracking):
    result = add_team_rebounding_features(games, team_games)
    profiles = tracking.copy()
    if profiles.duplicated(['player_id', 'season']).any():
        raise ValueError('Duplicate tracking player-season')
    profiles[TRACKING] = profiles[TRACKING].apply(pd.to_numeric)
    values = profiles[TRACKING].to_numpy(dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid tracking percentages')
    profiles['season'] = profiles.season.map(next_season)
    profiles['lag_reb_tracking_available'] = profiles[TRACKING].notna().all(axis=1).astype(int)
    profiles = profiles.rename(columns={'player_id': 'playerid', **{c: f'lag_track_{c}' for c in TRACKING}})
    result = result.merge(profiles[['playerid', 'season', 'lag_reb_tracking_available'] +
                                    [f'lag_track_{c}' for c in TRACKING]],
                           on=['playerid', 'season'], how='left', validate='many_to_one')
    result['lag_reb_tracking_available'] = result.lag_reb_tracking_available.fillna(0).astype(int)
    # These are fitted interaction terms, not an assumption that a player has
    # a rebound chance on every team miss or every rebound allowed.
    result['reb_offensive_matchup_interaction'] = result.lag_track_oreb_chance_pct * result.opp_reb_opponent_orb_prior10
    result['reb_defensive_matchup_interaction'] = result.lag_track_dreb_chance_pct * result.opp_reb_opponent_dreb_prior10
    return result
