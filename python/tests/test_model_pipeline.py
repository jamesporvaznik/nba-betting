"""Regression tests for chronology, pregame selection, and proxy evaluation."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from models.features.engineering import engineer_features, eligible_rows, add_rolling_averages
from models.evaluation.metrics import evaluate_over_under
from models.data.load import load_clusters, load_gamelogs


def game(player, day, points, minutes=30, season='2024-25', team='NYK', opponent='BOS'):
    return {'playerid': player, 'date': f'2025-01-{day:02d}', 'points': points,
            'rebounds': 5, 'assists': 3, 'minutes': minutes, 'season': season,
            'team': team, 'opponent': opponent, 'gameid': f'{day}_{team}_{opponent}',
            'homeaway': 'H'}


class HistoricalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([game(1, 1, 10), game(1, 3, 100),
                                   game(2, 1, 20), game(2, 3, 30),
                                   game(1, 5, 8), game(2, 5, 12)])

    def features(self, frame):
        with contextlib.redirect_stdout(io.StringIO()):
            return engineer_features(frame)

    def test_input_row_order_cannot_change_features(self):
        assert_frame_equal(self.features(self.frame), self.features(self.frame.sample(frac=1, random_state=3)))

    def test_current_and_future_outcomes_cannot_change_earlier_features(self):
        original = self.features(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.date >= '2025-01-03', ['points', 'minutes']] = [10000, 1]
        altered = self.features(changed)
        columns = ['points_avg5', 'points_avg_season', 'minutes_avg_season',
                   'opp_points_allowed_avg', 'prior_games_season']
        assert_frame_equal(original.loc[original.date <= '2025-01-03', columns],
                           altered.loc[altered.date <= '2025-01-03', columns])

    def test_opponent_uses_complete_prior_game_totals(self):
        result = self.features(self.frame)
        self.assertTrue(result.loc[result.date == '2025-01-01', 'opp_points_allowed_avg'].isna().all())
        self.assertTrue((result.loc[result.date == '2025-01-03', 'opp_points_allowed_avg'] == 30).all())
        self.assertTrue((result.loc[result.date == '2025-01-05', 'opp_points_allowed_avg'] == 80).all())

    def test_season_boundary_resets_histories(self):
        frame = pd.DataFrame([game(1, 1, 99, season='2024-25'), game(1, 3, 10, season='2025-26'),
                              game(1, 5, 20, season='2025-26')])
        result = add_rolling_averages(frame)
        self.assertTrue(pd.isna(result.iloc[1].points_avg_season))
        self.assertEqual(result.iloc[1].prior_games_season, 0)
        self.assertEqual(result.iloc[2].points_avg_season, 10)

    def test_low_current_minutes_do_not_exclude_eligible_game_or_erase_history(self):
        frame = pd.DataFrame([game(1, 1, 10, 30), game(1, 3, 2, 2), game(1, 5, 5, 20)])
        featured = self.features(frame)
        eligible = eligible_rows(featured, min_prior_games=1, min_prior_avg_minutes=10)
        self.assertEqual(eligible.date.dt.day.tolist(), [3, 5])
        self.assertEqual(eligible.iloc[1].minutes_avg_season, 16)
        self.assertEqual(eligible.iloc[1].points_avg5, 6)

    def test_future_appearances_do_not_make_early_games_eligible(self):
        short = self.features(self.frame[self.frame.date < '2025-01-05'])
        longer = self.features(self.frame)
        self.assertTrue(eligible_rows(short, min_prior_games=2).empty)
        self.assertEqual(eligible_rows(longer, min_prior_games=2).date.dt.day.tolist(), [5, 5])

    def test_duplicate_player_date_is_rejected(self):
        with self.assertRaises(ValueError):
            self.features(pd.concat([self.frame, self.frame.iloc[:1]]))


class ProxyMetricTests(unittest.TestCase):
    def test_player_specific_lines_and_ties(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            accuracy = evaluate_over_under([12, 8, 10, 12], [11, 9, 11, 10],
                                            lines=[10, 10, 10, 10])
        self.assertEqual(accuracy, 1)
        self.assertIn('scored 2/4', output.getvalue())

    def test_explicit_lines_required(self):
        with self.assertRaises(TypeError):
            evaluate_over_under([10], [12])

    def test_no_direction_is_not_reported_as_success(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(np.isnan(evaluate_over_under([10], [10], lines=[10])))


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.selected = None
        self.orders = []

    def table(self, name):
        return self

    def select(self, columns):
        self.selected = columns
        return self

    def in_(self, *args):
        return self

    def order(self, column):
        self.orders.append(column)
        return self

    def range(self, *args):
        return self

    def execute(self):
        return type('Result', (), {'data': self.rows})()


class LoaderTests(unittest.TestCase):
    def test_ninth_weight_loaded_in_one_select_argument(self):
        query = FakeQuery([{'player_id': 1, 'archetype_8_weight': .2}])
        with patch('models.data.load.get_client', return_value=query):
            result = load_clusters()
        self.assertIn('archetype_8_weight', query.selected.split(','))
        self.assertIn('archetype_8_weight', result.columns)
        self.assertEqual(query.orders, ['season', 'player_id'])

    def test_loader_retains_short_appearances(self):
        query = FakeQuery([game(1, 1, 2, 2), game(1, 3, 10, 30)])
        with patch('models.data.load.get_client', return_value=query), contextlib.redirect_stdout(io.StringIO()):
            result = load_gamelogs(['2024-25'])
        self.assertEqual(len(result), 2)
        self.assertEqual(query.orders, ['gamelogid'])


if __name__ == '__main__':
    unittest.main()
