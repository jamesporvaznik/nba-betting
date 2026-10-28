"""Prior-date opponent residuals from rolling baselines and lagged clusters.

Cluster numbers are used only to select peers inside a season. They are never
treated as a numeric feature or matched across independently fitted seasons.
"""

import numpy as np
import pandas as pd

from models.features.engineering import chronological
from models.features.playtype_matchup import next_season

STATS = ['points', 'rebounds', 'assists']
WINDOWS = [5, 10]


def matchup_columns(stat, window, scope):
    return [f'{scope}_{stat}_rolling{window}_delta',
            f'{scope}_{stat}_rolling{window}_minutes_delta',
            f'{scope}_decayed_appearances']


def cluster_groups(stat, profile_timing='previous'):
    available = 'retrospective_cluster_available' if profile_timing == 'completed_current' else 'lag_cluster_available'
    general = matchup_columns(stat, 10, 'general')
    cluster = matchup_columns(stat, 10, 'cluster') + [available]
    return {'baseline': [], 'cluster_availability': [available], 'opponent_rolling10': general,
            'cluster_rolling10': cluster,
            'opponent_and_cluster_rolling10': general + cluster,
            'opponent_and_cluster_rolling5': matchup_columns(stat, 5, 'general') +
             matchup_columns(stat, 5, 'cluster') + [available]}


def add_cluster_matchups(frame, assignments, half_life_days=30., shrinkage=20., profile_timing='previous'):
    """Leave the target player out; emit a whole date before adding its results.

Source appearances need ten prior games and >=10 average minutes over those
ten games. No current-game minutes/results determine source eligibility. All
histories reset by season. Missing clusters get zero effect/count plus a flag.
The decayed count is weight mass, not an independent-game sample size.
"""
    if profile_timing not in ['previous', 'completed_current']:
        raise ValueError('Unknown cluster timing')
    if half_life_days <= 0 or shrinkage <= 0:
        raise ValueError('Half life and shrinkage must be positive')
    frame = chronological(frame).reset_index(drop=True)
    if frame[['opponent', *STATS, 'minutes']].isna().any().any():
        raise ValueError('Missing opponent or outcome')
    if not np.isfinite(frame[STATS + ['minutes']].to_numpy(dtype=float)).all() or (frame[STATS + ['minutes']] < 0).any().any():
        raise ValueError('Invalid outcome')
    frame['opponent'] = frame.opponent.str.strip()
    profiles = assignments[['player_id', 'season', 'primary_archetype']].copy()
    if profiles.isna().any().any() or profiles.duplicated(['player_id', 'season']).any():
        raise ValueError('Missing or duplicate cluster assignment')
    labels = pd.to_numeric(profiles.primary_archetype)
    if ((labels < 0) | (labels % 1 != 0)).any() or not np.isfinite(labels).all():
        raise ValueError('Invalid cluster label')
    profiles['primary_archetype'] = labels.astype(int)
    if profile_timing == 'previous':
        profiles['season'] = profiles.season.map(next_season)
    profiles = profiles.rename(columns={'player_id': 'playerid', 'primary_archetype': 'lag_cluster'})
    frame = frame.merge(profiles, on=['playerid', 'season'], how='left', validate='many_to_one')
    frame['lag_cluster_available'] = frame.lag_cluster.notna().astype(int)
    groups = frame.groupby(['playerid', 'season'])
    prior_count = groups.cumcount()
    residuals, names = [], []
    source_ok = prior_count >= 10
    for window in WINDOWS:
        prior_minutes = groups.minutes.transform(lambda s: s.shift(1).rolling(window, min_periods=window).sum())
        if window == 10:
            source_ok &= prior_minutes / window >= 10
        for stat in STATS:
            prior_total = groups[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=window).sum())
            residuals.extend([frame[stat] - prior_total / window,
                              frame[stat] - frame.minutes * prior_total / prior_minutes.replace(0, np.nan)])
            names.extend([f'{stat}_rolling{window}_delta', f'{stat}_rolling{window}_minutes_delta'])
    residuals = np.column_stack(residuals)
    source_ok &= np.isfinite(residuals).all(axis=1)
    size = len(names)
    outputs = {scope: np.zeros((len(frame), size + 1)) for scope in ['general', 'cluster']}

    # State vectors contain weighted residual sums followed by weight mass.
    def value(state, key, date):
        entry = state.get(key)
        if entry is None:
            return np.zeros(size + 1)
        previous, vector = entry
        return vector * 2 ** (-(date - previous).days / half_life_days)

    for _, season in frame.groupby('season', sort=True):
        general, clustered, individual = {}, {}, {}
        for date, daily in season.groupby('date', sort=True):
            # No result on this date can affect any prediction on this date.
            for i in daily.index:
                row = frame.loc[i]
                own = value(individual, (row.opponent, row.playerid), date)
                for scope, state, key in [
                    ('general', general, row.opponent),
                    ('cluster', clustered, (row.opponent, row.lag_cluster)),
                ]:
                    if scope == 'cluster' and not row.lag_cluster_available:
                        continue
                    vector = value(state, key, date) - own
                    count = max(float(vector[-1]), 0.)
                    if count > 1e-10:
                        outputs[scope][i, :-1] = vector[:-1] / (count + shrinkage)
                        outputs[scope][i, -1] = count
            for i in daily.index:
                if not source_ok.iloc[i]:
                    continue
                row = frame.loc[i]
                addition = np.r_[residuals[i], 1.]
                destinations = [(general, row.opponent), (individual, (row.opponent, row.playerid))]
                if row.lag_cluster_available:
                    destinations.append((clustered, (row.opponent, row.lag_cluster)))
                for state, key in destinations:
                    state[key] = (date, value(state, key, date) + addition)
    for scope, values in outputs.items():
        columns = [f'{scope}_{name}' for name in names] + [f'{scope}_decayed_appearances']
        frame[columns] = values
    if profile_timing == 'completed_current':
        frame = frame.rename(columns={'lag_cluster': 'retrospective_cluster',
                                      'lag_cluster_available': 'retrospective_cluster_available'})
    return frame
