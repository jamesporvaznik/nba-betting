import unittest

import numpy as np

from models.evaluate_confidence import confidence_buckets, conformity_quantile, margin


class ConfidenceTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_correct_order_statistic(self):
        self.assertEqual(conformity_quantile(np.arange(1, 10), .8), 8)
        self.assertEqual(conformity_quantile(np.arange(1, 10), .9), 9)
        self.assertTrue(np.isinf(conformity_quantile([1, 2], .9)))

    def test_bad_scores_are_rejected(self):
        for scores, level in [([], .9), ([np.nan], .9), ([-1], .9), ([1], 1)]:
            with self.assertRaises(ValueError):
                conformity_quantile(scores, level)

    def test_direction_confidence_excludes_actual_pushes(self):
        buckets = confidence_buckets([12, 8, 10], [10, 10, 10], [.75, .25, .99])
        selected = next(b for b in buckets if b['confidence_range'] == '70%–80%')
        self.assertEqual(selected['count'], 2)
        self.assertEqual(selected['mean_confidence'], .75)
        self.assertEqual(selected['actual_direction_accuracy'], 1)
        self.assertEqual(sum(b['count'] for b in buckets), 2)

    def test_confidence_is_not_the_same_as_observed_accuracy(self):
        buckets = confidence_buckets([8, 12], [10, 10], [.75, .75])
        selected = next(b for b in buckets if b['count'])
        self.assertEqual(selected['mean_confidence'], .75)
        self.assertEqual(selected['actual_direction_accuracy'], .5)

    def test_margin_is_scaled_and_does_not_use_actual_outcome(self):
        np.testing.assert_array_equal(margin([20, 10], [15, 15], [5, 10]), [[1], [-.5]])


if __name__ == '__main__':
    unittest.main()
