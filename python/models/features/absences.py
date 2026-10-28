"""Conditional-on-known-absences features; no clusters or future box scores."""

import re

import numpy as np
import pandas as pd

from models.features.engineering import chronological

LOADS = ['minutes', 'used_possessions', 'usage_minutes', 'fga', 'assists', 'rebounds']


def positions(value):
    mapping = {'G': {'PG', 'SG'}, 'GUARD': {'PG', 'SG'},
               'F': {'SF', 'PF'}, 'FORWARD': {'SF', 'PF'}, 'CENTER': {'C'},
               **{p: {p} for p in ['PG', 'SG', 'SF', 'PF', 'C']}}
    result = set()
    for token in re.findall(r'[A-Z]+', str(value).upper()):
        result.update(mapping.get(token, set()))
    return frozenset(result)


def position_weight(left, right):
    if not left or not right:
        return .3
    return 1. if left == right else .6 if left & right else .3


def absence_groups():
    unweighted = [f'absent_{s}' for s in LOADS] + ['absent_count', 'absent_thin_profile_count']
    weighted = [f'position_absent_{s}' for s in LOADS] + ['absent_unknown_position_minutes']
    faded = [f'new_absence_{s}' for s in LOADS]
    return {'baseline': [], 'unweighted_absences': unweighted,
            'position_absences': unweighted + weighted,
            'position_and_adaptation': unweighted + weighted + faded}


def add_absence_features(frame, absences, player_positions, half_life=5.):
    if half_life <= 0:
        raise ValueError('Adaptation half life must be positive')
    frame = chronological(frame).reset_index(drop=True)
    frame['team'] = frame.team.str.strip()
    frame['opponent'] = frame.opponent.str.strip()
    absences = absences.copy()
    absences['date'] = pd.to_datetime(absences.date).dt.normalize()
    absences['team'] = absences.team.str.strip()
    absences['opponent'] = absences.opponent.str.strip()
    if absences.duplicated(['player_id', 'season', 'date']).any():
        raise ValueError('Duplicate absence date')
    if player_positions.duplicated('nba_api_id').any():
        raise ValueError('Duplicate player positions')
    position_map = dict(zip(player_positions.nba_api_id, player_positions.position.map(positions)))
    counts = ['minutes', 'fga', 'fta', 'turnovers', 'assists', 'rebounds']
    if not np.isfinite(frame[counts].to_numpy(float)).all() or (frame[counts] < 0).any().any():
        raise ValueError('Invalid box scores')
    frame['_used'] = frame.fga + .44 * frame.fta + frame.turnovers
    team_totals = frame.groupby(['season', 'team', 'date'])[['minutes', '_used']].transform('sum')
    frame['_team_minutes'] = team_totals.minutes
    frame['_team_used'] = team_totals._used
    calendars = {key: np.sort(group.date.unique()) for key, group in frame.groupby(['season', 'team'])}
    opponent_map = frame.groupby(['season', 'team', 'date']).opponent.agg(lambda s: frozenset(s)).to_dict()
    histories = {key: group.sort_values('date') for key, group in frame.groupby(['playerid', 'season', 'team'])}
    seen = set(zip(frame.playerid, frame.season, frame.date))
    episodes, enriched = {}, []
    for row in absences.sort_values(['player_id', 'season', 'team', 'stint_number', 'date']).itertuples():
        if (row.player_id, row.season, row.date) in seen:
            raise ValueError('Absence conflicts with a player appearance')
        key = row.season, row.team
        calendar = calendars.get(key)
        if calendar is None:
            raise ValueError('Absence has no team calendar')
        ordinal = int(np.searchsorted(calendar, row.date.to_datetime64()))
        if ordinal == len(calendar) or calendar[ordinal] != row.date.to_datetime64():
            raise ValueError('Absence has no team game date')
        if opponent_map[(row.season, row.team, row.date)] != {row.opponent}:
            raise ValueError('Absence opponent disagrees with game logs')
        first, last = pd.Timestamp(row.stint_first_game_date), pd.Timestamp(row.stint_last_game_date)
        if not first < row.date < last:
            raise ValueError('Absence outside stint interior')
        ekey = row.player_id, row.season, row.team, row.stint_number
        previous = episodes.get(ekey)
        if previous is None or ordinal != previous['ordinal'] + 1:
            history = histories.get((row.player_id, row.season, row.team))
            if history is None:
                raise ValueError('Missing pre-absence player history')
            prior = history[(history.date < row.date) & (history.date >= first)].tail(10)
            if prior.empty:
                raise ValueError('No profile before absence onset')
            minutes = float(prior.minutes.mean())
            usage = (float(prior._used.sum() * (prior._team_minutes.sum() / 5) /
                           (prior.minutes.sum() * prior._team_used.sum()))
                     if prior.minutes.sum() > 0 and prior._team_used.sum() > 0 else 0.)
            loads = [minutes, float(prior._used.mean()), usage * minutes,
                     float(prior.fga.mean()), float(prior.assists.mean()), float(prior.rebounds.mean())]
            previous = {'start': row.date, 'loads': loads, 'thin': int(len(prior) < 5)}
        previous['ordinal'] = ordinal
        episodes[ekey] = previous
        enriched.append({'season': row.season, 'team': row.team, 'date': row.date,
                         'absent_player': row.player_id, 'start': previous['start'],
                         'thin': previous['thin'], **dict(zip(LOADS, previous['loads']))})
    columns = absence_groups()['position_and_adaptation']
    output = pd.DataFrame(0., index=frame.index, columns=columns)
    if enriched:
        targets = frame[['playerid', 'season', 'team', 'date']].copy()
        targets['_row'] = frame.index
        pairs = targets.merge(pd.DataFrame(enriched), on=['season', 'team', 'date'], how='inner')
        if pairs.playerid.eq(pairs.absent_player).any():
            raise ValueError('Target is marked absent')
        weights = np.array([position_weight(position_map.get(p, frozenset()), position_map.get(a, frozenset()))
                            for p, a in zip(pairs.playerid, pairs.absent_player)])
        experienced = np.zeros(len(pairs))
        for (pid, season, team), group in pairs.groupby(['playerid', 'season', 'team']):
            dates = histories[(pid, season, team)].date.to_numpy()
            experienced[group.index] = (np.searchsorted(dates, group.date.to_numpy(), side='left') -
                                        np.searchsorted(dates, group.start.to_numpy(), side='left'))
        fade = 2 ** (-experienced / half_life)
        values = pd.DataFrame(index=pairs.index)
        for load in LOADS:
            values[f'absent_{load}'] = pairs[load].to_numpy()
            values[f'position_absent_{load}'] = pairs[load].to_numpy() * weights
            values[f'new_absence_{load}'] = pairs[load].to_numpy() * weights * fade
        values['absent_count'] = 1.
        values['absent_thin_profile_count'] = pairs.thin.to_numpy()
        unknown = np.array([not position_map.get(p) or not position_map.get(a)
                            for p, a in zip(pairs.playerid, pairs.absent_player)])
        values['absent_unknown_position_minutes'] = pairs.minutes.to_numpy() * unknown
        values['_row'] = pairs._row.to_numpy()
        summed = values.groupby('_row').sum()
        output.loc[summed.index, summed.columns] = summed
    frame[columns] = output
    return frame.drop(columns=['_used', '_team_minutes', '_team_used'])
