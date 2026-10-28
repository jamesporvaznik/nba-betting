import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from models.diagnose_predictions import add_segments, attach, bootstrap_mean, regression_summary, confidence_summary


class PredictionDiagnosticsTests(unittest.TestCase):
    def features(self):
        return pd.DataFrame({
            'playerid': [1, 2, 3], 'season': ['2024-25']*3,
            'date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'team': ['A']*3, 'opponent': ['B']*3, 'points': [9, 10, 30],
            'points_avg_season': [9, 15, 25], 'points_avg5': [8, 12, 20],
            'minutes_avg5': [15, 25, 35], 'minutes_avg_season': [21, 25, 29],
            'fga_prior5': [5, 10, 15], 'fga_priorseason': [10, 10, 10]})

    def test_segments_use_pregame_profiles_not_outcomes(self):
        original = self.features()
        before = add_segments(original)
        original['points'] = [1000, 0, 1000]
        after = add_segments(original)
        columns = ['scoring_level', 'recent_minutes', 'minutes_role', 'shot_role']
        assert_frame_equal(before[columns], after[columns])
        self.assertEqual(before.minutes_role.tolist(), ['falling (5+ min)', 'stable (<5 min change)', 'rising (5+ min)'])
        self.assertEqual(before.shot_role.tolist(), ['falling (25%+)', 'stable (<25% change)', 'rising (25%+)'])

    def test_paired_bootstrap_preserves_constant_difference(self):
        dates = pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02'])
        self.assertEqual(bootstrap_mean(dates, [-1, -1, -1], 100), [-1, -1])
        self.assertEqual(bootstrap_mean(dates[:1], [1]), [None, None])

    def test_join_rejects_unmatched_or_changed_saved_predictions(self):
        features = add_segments(self.features())
        predictions = features[['playerid', 'season', 'date', 'team', 'opponent', 'points']].copy()
        self.assertEqual(len(attach(predictions, features, 'points')), 3)
        for changes in [{'points': 1000}, {'team': 'C'}]:
            altered = predictions.copy()
            for column, value in changes.items():
                altered.loc[0, column] = value
            with self.assertRaises(ValueError):
                attach(altered, features, 'points')

    def test_regression_comparison_is_paired_on_identical_appearances(self):
        frame = self.features()
        frame['linear_baseline'] = frame.points + 2
        frame['linear_shot_volume'] = frame.points + 1
        result = regression_summary(frame, 'points')
        self.assertEqual(result['shots_minus_baseline_mae'], -1)
        self.assertEqual(result['mae_change_95pct_date_bootstrap'], [-1, -1])
        self.assertTrue(result['small_sample'])

    def test_confidence_compares_trend_and_calibrated_baseline_on_same_rows(self):
        frame = self.features()
        frame['pregame_average_proxy_line'] = frame.points_avg_season
        frame['probability_over_proxy_nonpush'] = [.5, .3, .7]
        frame['baseline_probability_over_proxy'] = [.5, .3, .3]
        frame['lower_90'] = 0
        frame['upper_90'] = 40
        result = confidence_summary(frame, 'points', False)
        self.assertEqual(result['nonpush_predictions'], 2)
        self.assertEqual(result['direction_accuracy'], 1)
        self.assertEqual(result['calibrated_baseline_accuracy'], .5)
        self.assertEqual(result['last5_comparison_predictions'], 2)
        self.assertEqual(result['last5_direction_accuracy'], .5)
        self.assertEqual(result['accuracy_gain_vs_last5'], .5)


if __name__ == '__main__':
    unittest.main()
