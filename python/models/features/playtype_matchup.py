"""Previous-season player style crossed with previous-season opponent defense."""

import numpy as np
import pandas as pd

PLAY_TYPES = ['Isolation', 'Transition', 'PRBallHandler', 'PRRollMan', 'Postup',
              'Spotup', 'Handoff', 'Cut', 'OffScreen', 'OffRebound', 'Misc']


def next_season(season):
    start, end = str(season).split('-')
    if int(end) != (int(start)+1) % 100:
        raise ValueError('Invalid NBA season')
    return f'{int(start)+1}-{(int(start)+2) % 100:02d}'


def playtype_columns(profile_timing='previous'):
    player = ['prior_player_profile_available'] + [f'prior_style_{t}' for t in PLAY_TYPES]
    defense = ['prior_opponent_profile_available'] + [f'prior_defense_{t}' for t in PLAY_TYPES]
    interaction = [f'prior_matchup_{t}' for t in PLAY_TYPES] + ['prior_style_weighted_defense']
    result = {'baseline': [], 'player_profile': player, 'opponent_profile': defense,
              'both_profiles': player + defense, 'interactions': player + defense + interaction}
    if profile_timing == 'completed_current':
        return {k: [c.replace('prior_', 'retrospective_', 1) for c in columns] for k, columns in result.items()}
    if profile_timing != 'previous':
        raise ValueError('Unknown profile timing')
    return result


def add_playtype_matchups(games, player_profiles, defensive_profiles, profile_timing='previous'):
    players, defense = player_profiles.copy(), defensive_profiles.copy()
    if players.duplicated(['season', 'player_id', 'play_type']).any():
        raise ValueError('Duplicate player playtype keys')
    if defense.duplicated(['season', 'opponent', 'play_type']).any():
        raise ValueError('Duplicate opponent playtype keys')
    if (set(players.play_type) | set(defense.play_type)) - set(PLAY_TYPES):
        raise ValueError('Unknown playtype')
    players['poss_pct'] = pd.to_numeric(players.poss_pct)
    defense[['ppp', 'possessions_defended']] = defense[['ppp', 'possessions_defended']].apply(pd.to_numeric)
    if players.poss_pct.isna().any() or not players.poss_pct.between(0, 1).all():
        raise ValueError('Invalid player possession proportions')
    if (not np.isfinite(defense[['ppp', 'possessions_defended']].to_numpy(dtype=float)).all()
            or (defense.ppp < 0).any() or (defense.possessions_defended <= 0).any()):
        raise ValueError('Invalid defensive profiles')
    if profile_timing == 'previous':
        players['season'] = players.season.map(next_season)
        defense['season'] = defense.season.map(next_season)
    elif profile_timing != 'completed_current':
        raise ValueError('Unknown profile timing')
    # Unreported types within an otherwise available player profile become zero;
    # wholly absent player profiles remain missing and receive an indicator.
    styles = players.pivot(index=['season', 'player_id'], columns='play_type', values='poss_pct').reindex(columns=PLAY_TYPES).fillna(0)
    styles.columns = [f'prior_style_{t}' for t in PLAY_TYPES]
    styles['prior_player_profile_available'] = 1
    styles = styles.reset_index().rename(columns={'player_id': 'playerid'})
    defense['weighted_ppp'] = defense.ppp * defense.possessions_defended
    league = defense.groupby(['season', 'play_type'])[['weighted_ppp', 'possessions_defended']].sum()
    league['league_ppp'] = league.weighted_ppp / league.possessions_defended
    defense = defense.merge(league[['league_ppp']], on=['season', 'play_type'], validate='many_to_one')
    defense['relative_ppp'] = defense.ppp-defense.league_ppp
    opponents = defense.pivot(index=['season', 'opponent'], columns='play_type', values='relative_ppp').reindex(columns=PLAY_TYPES)
    opponents.columns = [f'prior_defense_{t}' for t in PLAY_TYPES]
    # Team profiles are expected to have all eleven types. Missing coverage stays
    # explicit rather than dropping players or silently treating a team as average.
    opponents['prior_opponent_profile_available'] = opponents.notna().all(axis=1).astype(int)
    result = games.merge(styles, on=['season', 'playerid'], how='left', validate='many_to_one')
    result = result.merge(opponents.reset_index(), on=['season', 'opponent'], how='left', validate='many_to_one')
    for flag in ['prior_player_profile_available', 'prior_opponent_profile_available']:
        result[flag] = result[flag].fillna(0).astype(int)
    for t in PLAY_TYPES:
        result[f'prior_matchup_{t}'] = result[f'prior_style_{t}'] * result[f'prior_defense_{t}']
    result['prior_style_weighted_defense'] = result[[f'prior_matchup_{t}' for t in PLAY_TYPES]].sum(axis=1, min_count=len(PLAY_TYPES))
    if profile_timing == 'completed_current':
        result = result.rename(columns={c: c.replace('prior_', 'retrospective_', 1)
                                        for c in result.columns if c.startswith('prior_') and c not in games.columns})
    return result
