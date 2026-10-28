"""Previous-season shot distribution and opponent zone-defense interactions."""

import numpy as np
import pandas as pd

from models.features.playtype_matchup import next_season

ZONES = {'Restricted Area': ('rim', 2), 'In The Paint (Non-RA)': ('paint', 2),
         'Mid-Range': ('midrange', 2), 'Left Corner 3': ('left_corner3', 3),
         'Right Corner 3': ('right_corner3', 3), 'Above the Break 3': ('above_break3', 3)}


def shot_zone_columns(profile_timing='previous'):
    player = ['lag_zone_player_available'] + [f'lag_zone_share_{name}' for name, _ in ZONES.values()]
    defense = ['lag_zone_defense_available'] + [f'lag_zone_{measure}_{name}'
               for measure in ['fg_gap', 'frequency_gap'] for name, _ in ZONES.values()]
    interactions = [f'lag_zone_{measure}_{name}' for measure in ['scoring_interaction', 'frequency_interaction']
                    for name, _ in ZONES.values()] + ['lag_zone_scoring_matchup', 'lag_zone_frequency_alignment']
    result = {'baseline': [], 'player_profile': player, 'opponent_profile': defense,
              'both_profiles': player + defense, 'interactions': player + defense + interactions}
    if profile_timing == 'completed_current':
        return {k: [c.replace('lag_zone_', 'retrospective_zone_', 1) for c in cols] for k, cols in result.items()}
    if profile_timing != 'previous':
        raise ValueError('Unknown profile timing')
    return result


def add_shot_zone_matchups(games, player_profiles, defensive_profiles, profile_timing='previous'):
    if profile_timing not in ['previous', 'completed_current']:
        raise ValueError('Unknown profile timing')
    players, defense = player_profiles.copy(), defensive_profiles.copy()
    for frame, entity in [(players, 'player_id'), (defense, 'opponent')]:
        if frame.duplicated(['season', entity, 'shot_zone']).any():
            raise ValueError('Duplicate shot-zone profile keys')
        if set(frame.shot_zone) - set(ZONES):
            raise ValueError('Unknown shot zone')
        frame['frequency_pct'] = pd.to_numeric(frame.frequency_pct)
        if not np.isfinite(frame.frequency_pct).all() or not frame.frequency_pct.between(0, 1).all():
            raise ValueError('Invalid shot frequency')
        if profile_timing == 'previous':
            frame['season'] = frame.season.map(next_season)
    defense['fg_pct_allowed'] = pd.to_numeric(defense.fg_pct_allowed)
    if not np.isfinite(defense.fg_pct_allowed).all() or not defense.fg_pct_allowed.between(0, 1).all():
        raise ValueError('Invalid allowed field goal percentage')
    # Compare against the mean team zone percentage/frequency in the same prior
    # season. This is a mean of team rates, not an exact league pooled FG%.
    league = defense.groupby(['season', 'shot_zone'])[['fg_pct_allowed', 'frequency_pct']].transform('mean')
    defense['fg_gap'] = defense.fg_pct_allowed-league.fg_pct_allowed
    defense['frequency_gap'] = defense.frequency_pct-league.frequency_pct
    styles = players.pivot(index=['season', 'player_id'], columns='shot_zone', values='frequency_pct').reindex(columns=ZONES).fillna(0)
    styles.columns = [f'lag_zone_share_{name}' for name, _ in ZONES.values()]
    styles['lag_zone_player_available'] = 1
    styles = styles.reset_index().rename(columns={'player_id': 'playerid'})
    result = games.merge(styles, on=['season', 'playerid'], how='left', validate='many_to_one')
    for measure in ['fg_gap', 'frequency_gap']:
        wide = defense.pivot(index=['season', 'opponent'], columns='shot_zone', values=measure).reindex(columns=ZONES)
        wide.columns = [f'lag_zone_{measure}_{name}' for name, _ in ZONES.values()]
        if measure == 'fg_gap':
            wide['lag_zone_defense_available'] = wide.notna().all(axis=1).astype(int)
        result = result.merge(wide.reset_index(), on=['season', 'opponent'], how='left', validate='many_to_one')
    for flag in ['lag_zone_player_available', 'lag_zone_defense_available']:
        result[flag] = result[flag].fillna(0).astype(int)
    for name, point_value in ZONES.values():
        result[f'lag_zone_scoring_interaction_{name}'] = point_value * result[f'lag_zone_share_{name}'] * result[f'lag_zone_fg_gap_{name}']
        result[f'lag_zone_frequency_interaction_{name}'] = result[f'lag_zone_share_{name}'] * result[f'lag_zone_frequency_gap_{name}']
    result['lag_zone_scoring_matchup'] = result[[f'lag_zone_scoring_interaction_{n}' for n, _ in ZONES.values()]].sum(axis=1, min_count=len(ZONES))
    result['lag_zone_frequency_alignment'] = result[[f'lag_zone_frequency_interaction_{n}' for n, _ in ZONES.values()]].sum(axis=1, min_count=len(ZONES))
    if profile_timing == 'completed_current':
        result = result.rename(columns={c: c.replace('lag_zone_', 'retrospective_zone_', 1)
                                        for c in result if c.startswith('lag_zone_') and c not in games.columns})
    return result
