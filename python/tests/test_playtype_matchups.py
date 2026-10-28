import unittest

import numpy as np
import pandas as pd

from models.features.playtype_matchup import add_playtype_matchups, next_season, PLAY_TYPES, playtype_columns
from models.experiment_playtypes import chronological_folds


class PlaytypeMatchupTests(unittest.TestCase):
    def inputs(self):
        games = pd.DataFrame({'playerid': [1, 2], 'season': ['2023-24']*2, 'opponent': ['BOS']*2})
        players = pd.DataFrame([{'player_id': 1, 'season': '2022-23', 'play_type': 'Isolation', 'poss_pct': .4},
                                {'player_id': 1, 'season': '2022-23', 'play_type': 'Spotup', 'poss_pct': .6}])
        defense = pd.DataFrame([{'opponent': opponent, 'season': '2022-23', 'play_type': t,
                                 'ppp': ppp, 'possessions_defended': 100}
                                for opponent, ppp in [('BOS', .8), ('NYK', 1.2)] for t in PLAY_TYPES])
        return games, players, defense

    def test_previous_season_profiles_and_weighted_interaction(self):
        games, players, defense = self.inputs()
        result = add_playtype_matchups(games, players, defense)
        self.assertAlmostEqual(result.iloc[0].prior_defense_Isolation, -.2)
        self.assertAlmostEqual(result.iloc[0].prior_matchup_Isolation, -.08)
        self.assertAlmostEqual(result.iloc[0].prior_style_weighted_defense, -.2)
        self.assertEqual(result.iloc[0].prior_style_Cut, 0)

    def test_missing_player_profiles_remain_missing_with_indicator(self):
        result = add_playtype_matchups(*self.inputs())
        self.assertEqual(result.iloc[1].prior_player_profile_available, 0)
        self.assertTrue(np.isnan(result.iloc[1].prior_style_Isolation))
        self.assertTrue(np.isnan(result.iloc[1].prior_style_weighted_defense))

    def test_current_season_profiles_never_attach_to_current_games(self):
        games, players, defense = self.inputs()
        current = players.copy()
        current['season'] = '2023-24'
        current['poss_pct'] = [.99, .01]
        result = add_playtype_matchups(games, pd.concat([players, current]), defense)
        self.assertEqual(result.iloc[0].prior_style_Isolation, .4)

    def test_duplicate_profile_keys_rejected(self):
        games, players, defense = self.inputs()
        with self.assertRaises(ValueError):
            add_playtype_matchups(games, pd.concat([players, players.iloc[:1]]), defense)

    def test_completed_current_mode_is_explicitly_retrospective(self):
        games, players, defense = self.inputs()
        games['prior_games_season'] = [20, 21]
        players['season'] = '2023-24'
        defense['season'] = '2023-24'
        result = add_playtype_matchups(games, players, defense, 'completed_current')
        self.assertEqual(result.iloc[0].retrospective_style_Isolation, .4)
        self.assertNotIn('prior_style_Isolation', result.columns)
        self.assertEqual(result.prior_games_season.tolist(), [20, 21])
        self.assertTrue(all(c.startswith('retrospective_') for c in playtype_columns('completed_current')['interactions']))
        with self.assertRaises(ValueError):
            playtype_columns('unknown')

    def test_season_mapping(self):
        self.assertEqual(next_season('2022-23'), '2023-24')
        self.assertEqual(next_season('1999-00'), '2000-01')
        with self.assertRaises(ValueError):
            next_season('2022-24')

    def test_training_profiles_exist_before_each_chronological_fold(self):
        frame = pd.DataFrame({'season': ['2022-23', '2023-24', '2023-24', '2024-25'],
                              'date': pd.to_datetime(['2023-01-01', '2024-01-31', '2024-02-01', '2025-01-01'])})
        folds = chronological_folds(frame)
        self.assertEqual(folds[0][1].date.dt.day.tolist(), [31])
        self.assertEqual(folds[0][2].date.dt.day.tolist(), [1])
        self.assertEqual(set(folds[1][1].season), {'2023-24'})
        for _, tr, va in folds:
            self.assertLess(tr.date.max(), va.date.min())


if __name__ == '__main__':
    unittest.main()
