"""Fixed feature specification selected by chronological development experiments."""

from contextlib import redirect_stdout
import io

import pandas as pd

from models.features.absences import absence_groups, add_absence_features
from models.features.engineering import engineer_features, get_feature_cols
from models.features.matchup import add_matchup_features, experimental_groups
from models.features.offense import add_offense_features, offense_groups
from models.features.rebounding import add_team_rebounding_features, rebound_columns

STATS = ['points', 'rebounds', 'assists']


def selected_feature_cols(stat):
    if stat not in STATS:
        raise ValueError(f'Unsupported target: {stat}')
    columns = get_feature_cols(stat) + offense_groups(stat)['shot_volume']
    columns += absence_groups()['position_and_adaptation']
    if stat in ['rebounds', 'assists']:
        columns += experimental_groups(stat)['opportunity']
    if stat == 'rebounds':
        columns += rebound_columns()['team_rebounding']
    return columns


def build_selected_features(snapshot):
    """Engineer complete histories; callers apply pregame eligibility afterward."""
    raw = pd.DataFrame(snapshot['players'])
    if raw.empty:
        raise ValueError('No game logs were supplied')
    with redirect_stdout(io.StringIO()):
        frame = add_offense_features(add_matchup_features(engineer_features(raw), pd.DataFrame(snapshot['teams'])))
        frame = add_team_rebounding_features(frame, pd.DataFrame(snapshot['teams']))
        frame = add_absence_features(frame, pd.DataFrame(snapshot['absences']), pd.DataFrame(snapshot['positions']))
    for stat in STATS:
        frame[f'{stat}_scale'] = frame.groupby(['playerid', 'season'])[stat].transform(
            lambda x: x.shift(1).rolling(10, min_periods=2).std()).clip(lower=1)
    return frame
