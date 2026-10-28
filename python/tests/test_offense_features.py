import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from models.features.offense import add_offense_features, offense_groups


class OffenseFeatureTests(unittest.TestCase):
    def inputs(self):
        return pd.DataFrame([
            dict(playerid=player, season='2023-24', date=f'2024-01-{day:02d}',
                 gameid=f'{day}_A_B', team='A', minutes=20, fga=shots, fta=0,
                 **{'3PA': shots // 2}, assists=2, turnovers=1)
            for day in [1, 3, 5] for player, shots in [(1, 10), (2, 30)]])

    def test_shares_use_complete_team_games_then_shift(self):
        result = add_offense_features(self.inputs())
        player = result[result.playerid == 1].reset_index(drop=True)
        self.assertTrue(pd.isna(player.iloc[0].fga_prior5))
        self.assertEqual(player.iloc[1].fga_team_share_prior10, .25)
        self.assertEqual(player.iloc[1].fga_per_minute_prior10, .5)
        self.assertEqual(player.iloc[1].fta_team_share_prior10, 0)

    def test_current_and_future_teammate_outcomes_cannot_change_prior_inputs(self):
        original = self.inputs()
        before = add_offense_features(original)
        changed = original.copy()
        changed.loc[changed.date >= '2024-01-03', ['fga', 'fta', 'assists', 'minutes']] = [1000, 100, 20, 2]
        after = add_offense_features(changed)
        columns = offense_groups('points')['shot_volume'] + offense_groups('points')['involvement']
        assert_frame_equal(before.loc[before.date <= '2024-01-03', columns],
                           after.loc[after.date <= '2024-01-03', columns])

    def test_input_order_does_not_change_features(self):
        frame = self.inputs()
        assert_frame_equal(add_offense_features(frame), add_offense_features(frame.sample(frac=1)))

    def test_season_reset(self):
        frame = self.inputs()
        frame.loc[frame.date >= '2024-01-03', 'season'] = '2024-25'
        result = add_offense_features(frame)
        self.assertTrue(result.loc[result.date == '2024-01-03', 'fga_priorseason'].isna().all())

    def test_trade_uses_actual_team_for_each_completed_game(self):
        frame = self.inputs()
        # Player 1 moves to C with a 10-attempt teammate, instead of A (30).
        frame.loc[(frame.date == '2024-01-03') & (frame.playerid == 1), ['team', 'gameid']] = ['C', '3_C_B']
        teammate = frame[(frame.date == '2024-01-03') & (frame.playerid == 1)].copy()
        teammate['playerid'] = 3
        result = add_offense_features(pd.concat([frame, teammate], ignore_index=True))
        final = result[(result.playerid == 1) & (result.date == '2024-01-05')].iloc[0]
        self.assertEqual(final.fga_team_share_prior10, (.25 + .5) / 2)


if __name__ == '__main__':
    unittest.main()
