import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from models.features.engineering import engineer_features
from models.features.matchup import add_matchup_features, experimental_groups
from models.experiment_features import paired_date_interval


class MatchupFeaturesTests(unittest.TestCase):
    def inputs(self):
        players, teams = [], []
        for day in [1, 3, 5]:
            date = f'2024-01-{day:02d}'
            players.append(dict(playerid=1, date=date, season='2023-24', team='NYK',
                                opponent='BOS', gameid=f'{day}_NYK_BOS', homeaway='H',
                                points=10, rebounds=4, assists=2, minutes=20))
            for team in ['NYK', 'BOS']:
                teams.append(dict(date=date, season='2023-24', team=team, team_pace=100+day,
                                  def_rating=110+day, opponent_fgm=40, opponent_fga=80,
                                  opponent_3pm=10, opponent_3pa=25, opponent_fta=20,
                                  opponent_pts=100+day, opponent_trb=40, opponent_ast=20))
        return pd.DataFrame(players), pd.DataFrame(teams)

    def build(self, players, teams):
        with contextlib.redirect_stdout(io.StringIO()):
            return add_matchup_features(engineer_features(players), teams)

    def test_prior_only_team_and_player_features(self):
        players, teams = self.inputs()
        result = self.build(players, teams)
        self.assertTrue(pd.isna(result.iloc[0].own_pace_prior_season))
        self.assertEqual(result.iloc[1].expected_pace_proxy, 101)
        self.assertEqual(result.iloc[2].opp_def_rating_prior_season, 112)
        self.assertEqual(result.iloc[2].opp_fg_pct_allowed_prior, .5)
        self.assertEqual(result.iloc[2].points_per_minute_10, .5)
        self.assertEqual(result.iloc[2].points_opportunity_proxy, 10)

    def test_current_and_future_outcomes_do_not_change_prior_features(self):
        players, teams = self.inputs()
        before = self.build(players, teams)
        players.loc[players.date >= '2024-01-03', ['points', 'minutes']] = [1000, 1]
        numeric = teams.select_dtypes('number').columns
        teams.loc[teams.date >= '2024-01-03', numeric] *= 20
        after = self.build(players, teams)
        columns = experimental_groups('points')['combined']
        assert_frame_equal(before.loc[:1, columns], after.loc[:1, columns])

    def test_shuffling_team_and_player_rows_does_not_change_results(self):
        players, teams = self.inputs()
        assert_frame_equal(self.build(players, teams), self.build(players.sample(frac=1), teams.sample(frac=1)))

    def test_duplicate_team_date_is_rejected(self):
        players, teams = self.inputs()
        with self.assertRaises(ValueError):
            self.build(players, pd.concat([teams, teams.iloc[:1]]))

    def test_both_team_and_player_histories_reset_by_season(self):
        players, teams = self.inputs()
        players.loc[players.date >= '2024-01-03', 'season'] = '2024-25'
        teams.loc[teams.date >= '2024-01-03', 'season'] = '2024-25'
        result = self.build(players, teams)
        self.assertTrue(pd.isna(result.iloc[1].own_pace_prior_season))
        self.assertTrue(pd.isna(result.iloc[1].points_per_minute_season))

    def test_date_block_bootstrap_uses_paired_errors(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        bounds = paired_date_interval(dates, np.array([3., 4., 5.]), np.array([2., 3., 4.]), repeats=100)
        self.assertEqual(bounds, [-1., -1.])


if __name__ == '__main__':
    unittest.main()
