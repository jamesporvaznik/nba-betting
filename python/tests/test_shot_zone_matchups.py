import unittest

import numpy as np
import pandas as pd

from models.features.shot_zone_matchup import ZONES, add_shot_zone_matchups, shot_zone_columns


class ShotZoneMatchupTests(unittest.TestCase):
    def inputs(self):
        games = pd.DataFrame({'playerid': [1, 2], 'season': ['2023-24']*2, 'opponent': ['BOS']*2})
        players = pd.DataFrame([
            {'player_id': 1, 'season': '2022-23', 'shot_zone': 'Restricted Area', 'frequency_pct': .25},
            {'player_id': 1, 'season': '2022-23', 'shot_zone': 'Above the Break 3', 'frequency_pct': .75}])
        defense = pd.DataFrame([
            {'opponent': team, 'season': '2022-23', 'shot_zone': zone, 'fg_pct_allowed': pct,
             'frequency_pct': (.5 if zone == 'Restricted Area' else .1) if team == 'BOS'
             else (0 if zone == 'Restricted Area' else .2)}
            for team, pct in [('BOS', .4), ('NYK', .6)] for zone in ZONES])
        return games, players, defense

    def test_zone_interaction_uses_previous_profile_and_correct_point_value(self):
        result = add_shot_zone_matchups(*self.inputs())
        self.assertAlmostEqual(result.iloc[0].lag_zone_scoring_interaction_rim, -.05)
        self.assertAlmostEqual(result.iloc[0].lag_zone_scoring_interaction_above_break3, -.225)
        self.assertAlmostEqual(result.iloc[0].lag_zone_scoring_matchup, -.275)
        self.assertAlmostEqual(result.iloc[0].lag_zone_frequency_alignment, .025)
        self.assertEqual(result.iloc[0].lag_zone_share_midrange, 0)

    def test_absent_profile_keeps_player_and_missing_flag(self):
        result = add_shot_zone_matchups(*self.inputs())
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[1].lag_zone_player_available, 0)
        self.assertTrue(np.isnan(result.iloc[1].lag_zone_share_rim))
        self.assertTrue(np.isnan(result.iloc[1].lag_zone_scoring_matchup))

    def test_current_season_profiles_cannot_change_current_game_features(self):
        games, players, defense = self.inputs()
        current_players = players.assign(season='2023-24', frequency_pct=[.99, .01])
        current_defense = defense.assign(season='2023-24', fg_pct_allowed=.99)
        before = add_shot_zone_matchups(games, players, defense)
        after = add_shot_zone_matchups(games, pd.concat([players, current_players]), pd.concat([defense, current_defense]))
        pd.testing.assert_frame_equal(before, after)

    def test_duplicate_or_invalid_profiles_rejected(self):
        games, players, defense = self.inputs()
        with self.assertRaises(ValueError):
            add_shot_zone_matchups(games, pd.concat([players, players.iloc[:1]]), defense)
        with self.assertRaises(ValueError):
            add_shot_zone_matchups(games, players, defense.assign(fg_pct_allowed=np.inf))
        with self.assertRaises(ValueError):
            add_shot_zone_matchups(games, players.assign(frequency_pct=25), defense)

    def test_absent_opponent_zone_stays_missing(self):
        games, players, defense = self.inputs()
        defense = defense[~((defense.opponent == 'BOS') & (defense.shot_zone == 'Mid-Range'))]
        result = add_shot_zone_matchups(games, players, defense)
        self.assertEqual(result.iloc[0].lag_zone_defense_available, 0)
        self.assertTrue(np.isnan(result.iloc[0].lag_zone_scoring_matchup))

    def test_current_profiles_attach_without_relabeling_original_game_fields(self):
        games, players, defense = self.inputs()
        games['prior_games_season'] = [20, 21]
        result = add_shot_zone_matchups(games, players.assign(season='2023-24'),
                                        defense.assign(season='2023-24'), 'completed_current')
        self.assertEqual(result.iloc[0].retrospective_zone_share_rim, .25)
        self.assertNotIn('lag_zone_share_rim', result)
        self.assertEqual(result.prior_games_season.tolist(), [20, 21])
        self.assertTrue(all(c.startswith('retrospective_zone_') for c in shot_zone_columns('completed_current')['interactions']))
        with self.assertRaises(ValueError):
            add_shot_zone_matchups(games, players, defense, 'unknown')

    def test_timing_modes_are_equivalent_when_only_profile_season_is_shifted(self):
        games, players, defense = self.inputs()
        previous = add_shot_zone_matchups(games, players, defense)
        current = add_shot_zone_matchups(games, players.assign(season='2023-24'),
                                         defense.assign(season='2023-24'), 'completed_current')
        current = current.rename(columns={c: c.replace('retrospective_zone_', 'lag_zone_', 1) for c in current if c.startswith('retrospective_zone_')})
        pd.testing.assert_frame_equal(previous, current)


if __name__ == '__main__':
    unittest.main()
