import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from models.experiment_minutes import production, rate_training_rows, direct_predictions


class MinutesExperimentTests(unittest.TestCase):
    def test_production_uses_predicted_components_and_is_nonnegative(self):
        np.testing.assert_array_equal(production([30, -2, 10], [.5, .5, -.1]), [15, 0, 0])
        with self.assertRaises(ValueError):
            production([30], [np.nan])
        with self.assertRaises(ValueError):
            production([30, 20], [.5])

    def test_rate_training_handles_zero_minutes_and_weights_exposure(self):
        frame = pd.DataFrame({'minutes': [0, 2, 20], 'points': [0, 4, 10]})
        usable, target, weights = rate_training_rows(frame, 'points')
        self.assertEqual(usable.index.tolist(), [1, 2])
        np.testing.assert_array_equal(target, [2, .5])
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[1]/weights[0], 10)

    def test_saved_direct_predictions_match_by_keys_not_row_position(self):
        frame = pd.DataFrame({'playerid': [1, 2], 'season': ['2024-25']*2,
                              'date': pd.to_datetime(['2025-01-01']*2), 'team': ['A']*2,
                              'opponent': ['B']*2, 'points': [10, 20]})
        saved = frame.assign(linear_shot_volume=[11, 21], xgboost_shot_volume=[12, 22])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved.iloc[::-1].to_csv(root/'predictions_2024-25_points.csv', index=False)
            predictions = direct_predictions(root, '2024-25', 'points', frame)
            np.testing.assert_array_equal(predictions['linear'], [11, 21])
            saved.loc[0, 'points'] = 100
            saved.to_csv(root/'predictions_2024-25_points.csv', index=False)
            with self.assertRaises(ValueError):
                direct_predictions(root, '2024-25', 'points', frame)


if __name__ == '__main__':
    unittest.main()
