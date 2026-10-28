import unittest

import numpy as np
import pandas as pd

from models.features.rebounding import add_rebounding_features, rebound_columns, TRACKING


class ReboundingFeatureTests(unittest.TestCase):
    def inputs(self):
        games = pd.DataFrame([{'playerid': 1, 'season': '2023-24', 'date': f'2024-01-{d:02d}',
                               'team': 'A', 'opponent': 'B'} for d in [1, 3, 5]])
        teams = []
        for d in [1, 3, 5]:
            common = {'season': '2023-24', 'date': f'2024-01-{d:02d}'}
            teams.append({**common, 'team': 'A', 'opponent_abbrev': 'B', 'team_orb': 10, 'team_trb': 40,
                          'opponent_orb': 5, 'opponent_trb': 35, 'opponent_fga': 82, 'opponent_fgm': 42})
            teams.append({**common, 'team': 'B', 'opponent_abbrev': 'A', 'team_orb': 5, 'team_trb': 35,
                          'opponent_orb': 10, 'opponent_trb': 40, 'opponent_fga': 90, 'opponent_fgm': 45})
        tracking = pd.DataFrame([{'player_id': 1, 'season': '2022-23', 'reb_chance_pct': .6,
                                  'dreb_chance_pct': .7, 'oreb_chance_pct': .3, 'reb_contest_pct': .2}])
        return games, pd.DataFrame(teams), tracking

    def test_opponent_orb_means_rebounds_allowed_by_upcoming_opponent(self):
        result = add_rebounding_features(*self.inputs())
        row = result.iloc[1]
        self.assertTrue(pd.isna(result.iloc[0].opp_reb_opponent_orb_prior10))
        self.assertEqual(row.opp_reb_opponent_orb_prior10, 10)
        self.assertEqual(row.opp_reb_team_orb_prior10, 5)
        self.assertEqual(row.opp_reb_team_fg_misses_prior10, 40)
        self.assertEqual(row.opp_reb_opponent_fg_misses_prior10, 45)
        self.assertAlmostEqual(row.opp_reb_defensive_reb_rate_prior10, .75)
        self.assertAlmostEqual(row.own_reb_offensive_reb_rate_prior10, .25)
        self.assertEqual(row.reb_offensive_matchup_interaction, 3)

    def test_current_and_future_team_outcomes_do_not_change_earlier_features(self):
        games, teams, tracking = self.inputs()
        before = add_rebounding_features(games, teams, tracking)
        columns = teams.select_dtypes('number').columns
        teams.loc[teams.date >= '2024-01-03', columns] *= 2
        after = add_rebounding_features(games, teams, tracking)
        cols = rebound_columns()['interactions']
        pd.testing.assert_frame_equal(before.loc[:1, cols], after.loc[:1, cols])

    def test_current_tracking_does_not_replace_lagged_profile_and_rookies_stay(self):
        games, teams, tracking = self.inputs()
        current = tracking.assign(season='2023-24', **{c: .99 for c in TRACKING})
        games.loc[2, 'playerid'] = 2
        result = add_rebounding_features(games, teams, pd.concat([tracking, current]))
        self.assertEqual(result.iloc[1].lag_track_reb_chance_pct, .6)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[2].lag_reb_tracking_available, 0)
        self.assertTrue(np.isnan(result.iloc[2].lag_track_reb_chance_pct))

    def test_new_season_resets_team_history(self):
        games, teams, tracking = self.inputs()
        games.loc[games.date >= '2024-01-03', 'season'] = '2024-25'
        teams.loc[teams.date >= '2024-01-03', 'season'] = '2024-25'
        result = add_rebounding_features(games, teams, tracking)
        self.assertTrue(pd.isna(result.iloc[1].opp_reb_opponent_orb_prior10))

    def test_missing_mirror_and_invalid_tracking_rejected(self):
        games, teams, tracking = self.inputs()
        with self.assertRaises(ValueError):
            add_rebounding_features(games, teams.iloc[1:], tracking)
        with self.assertRaises(ValueError):
            add_rebounding_features(games, teams, tracking.assign(reb_chance_pct=1.5))


if __name__ == '__main__':
    unittest.main()
