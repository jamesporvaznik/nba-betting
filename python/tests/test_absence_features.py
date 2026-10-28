import unittest

import numpy as np
import pandas as pd

from models.features.absences import add_absence_features, absence_groups, positions, position_weight


class AbsenceFeatureTests(unittest.TestCase):
    def inputs(self):
        games, absent = [], []
        for day in range(1, 23):
            date = f'2024-01-{day:02d}'
            for player in [1, 2]:
                if player == 2 and 6 <= day <= 20:
                    absent.append({'player_id': 2, 'season': '2023-24', 'team': 'A', 'date': date,
                                   'opponent': 'B', 'stint_number': 1,
                                   'stint_first_game_date': '2024-01-01', 'stint_last_game_date': '2024-01-22'})
                    continue
                games.append({'playerid': player, 'season': '2023-24', 'team': 'A', 'opponent': 'B',
                              'date': date, 'minutes': 20., 'fga': 8., 'fta': 0., 'turnovers': 2.,
                              'points': 10., 'assists': 4., 'rebounds': 5.})
        pos = pd.DataFrame([{'nba_api_id': 1, 'position': 'PG'}, {'nba_api_id': 2, 'position': 'PG'}])
        return pd.DataFrame(games), pd.DataFrame(absent), pos

    def test_minutes_usage_position_and_adaptation_hand_calculation(self):
        result = add_absence_features(*self.inputs())
        player = result[result.playerid.eq(1)].set_index('date')
        row = player.loc['2024-01-06']
        self.assertEqual(row.absent_minutes, 20)
        self.assertEqual(row.position_absent_minutes, 20)
        self.assertEqual(row.absent_used_possessions, 10)
        self.assertAlmostEqual(row.absent_usage_minutes, 4)
        self.assertEqual(row.new_absence_minutes, 20)
        self.assertAlmostEqual(player.loc['2024-01-11'].new_absence_minutes, 10)
        self.assertAlmostEqual(player.loc['2024-01-16'].new_absence_minutes, 5)
        self.assertEqual(player.loc['2024-01-20'].absent_minutes, 20)
        self.assertEqual(player.loc['2024-01-21'].absent_count, 0)

    def test_position_weights_and_unknown(self):
        self.assertEqual(position_weight(positions('PG'), positions('PG')), 1)
        self.assertEqual(position_weight(positions('Guard'), positions('PG')), .6)
        self.assertEqual(position_weight(positions('Guard-Forward'), positions('C')), .3)
        games, absent, pos = self.inputs()
        pos.loc[pos.nba_api_id.eq(2), 'position'] = None
        result = add_absence_features(games, absent, pos)
        row = result[result.date.eq('2024-01-06')].iloc[0]
        self.assertEqual(row.position_absent_minutes, 6)
        self.assertEqual(row.absent_unknown_position_minutes, 20)

    def test_current_future_results_cannot_change_absence_profiles(self):
        games, absent, pos = self.inputs()
        before = add_absence_features(games, absent, pos)
        games.loc[games.date >= '2024-01-06', ['minutes', 'fga', 'fta', 'turnovers', 'assists', 'rebounds']] *= 5
        after = add_absence_features(games, absent, pos)
        columns = absence_groups()['position_and_adaptation']
        pd.testing.assert_frame_equal(before[columns], after[columns])

    def test_episode_resets_on_return_and_uses_new_prior_profile(self):
        games, absent, pos = self.inputs()
        # Player 2 returns on day 11, then begins another absence on day 12.
        absent = absent[absent.date.ne('2024-01-11')]
        returned = games.iloc[0].to_dict()
        returned.update(playerid=2, date='2024-01-11', minutes=40.)
        games = pd.concat([games, pd.DataFrame([returned])], ignore_index=True)
        result = add_absence_features(games, absent, pos)
        row = result[result.playerid.eq(1) & result.date.eq('2024-01-12')].iloc[0]
        self.assertAlmostEqual(row.absent_minutes, 140 / 6)
        self.assertAlmostEqual(row.new_absence_minutes, row.position_absent_minutes)

    def test_reordering_is_stable_and_target_outcome_not_used(self):
        games, absent, pos = self.inputs()
        before = add_absence_features(games, absent, pos)
        after = add_absence_features(games.sample(frac=1, random_state=1), absent.sample(frac=1, random_state=2), pos)
        pd.testing.assert_frame_equal(before, after)

    def test_conflicting_and_duplicate_absences_rejected(self):
        games, absent, pos = self.inputs()
        for invalid in [pd.concat([absent, absent.iloc[:1]]), absent.assign(player_id=1)]:
            with self.assertRaises(ValueError):
                add_absence_features(games, invalid, pos)

    def test_adaptation_counts_target_appearances_not_calendar_days(self):
        games, absent, pos = self.inputs()
        # Target also misses five dates: no adaptation appearances accrue there.
        games = games[~(games.playerid.eq(1) & games.date.between('2024-01-06', '2024-01-10'))]
        # Another teammate preserves the team game calendar.
        fillers = []
        for day in range(6, 11):
            row = games.iloc[0].to_dict()
            row.update(playerid=3, date=f'2024-01-{day:02d}')
            fillers.append(row)
        result = add_absence_features(pd.concat([games, pd.DataFrame(fillers)]), absent, pos)
        row = result[result.playerid.eq(1) & result.date.eq('2024-01-11')].iloc[0]
        self.assertEqual(row.new_absence_minutes, 20)


if __name__ == '__main__':
    unittest.main()
