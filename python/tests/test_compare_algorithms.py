import unittest

import numpy as np

from models.compare_algorithms import blend_prediction, candidates, choose_candidate


class AlgorithmComparisonTests(unittest.TestCase):
    def rows(self):
        return [{'candidate': candidate, 'season': season, 'mae': score}
                for candidate, scores in [('linear', [2., 4.]), ('a', [1.8, 4.]), ('b', [2., 3.7])]
                for season, score in zip(['2023-24', '2024-25'], scores)]

    def test_selection_uses_equal_relative_fold_error(self):
        # A improves the first fold by 10%; B improves the second by 7.5%.
        # Raw pooled MAE would prefer B, but our prespecified relative rule picks A.
        self.assertEqual(choose_candidate(self.rows()), 'a')
        self.assertEqual(choose_candidate(self.rows(), ['linear', 'b']), 'b')

    def test_selection_rejects_test_rows_and_incomplete_candidates(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            choose_candidate(rows + [{'candidate': 'a', 'season': '2025-26', 'mae': 0}])
        with self.assertRaises(ValueError):
            choose_candidate(rows[:-1])

    def test_blend_endpoints_and_mixture(self):
        np.testing.assert_array_equal(blend_prediction([2, 6], [6, 2], 0), [2, 6])
        np.testing.assert_array_equal(blend_prediction([2, 6], [6, 2], 1), [6, 2])
        np.testing.assert_array_equal(blend_prediction([2, 6], [6, 2], .25), [3, 5])
        with self.assertRaises(ValueError):
            blend_prediction([1], [1], 1.1)

    def test_grid_has_both_losses_and_original_reference(self):
        grid = candidates()
        self.assertEqual(len(grid), len({c['id'] for c in grid}))
        self.assertEqual({c['family'] for c in grid}, {'linear', 'xgb_rmse', 'xgb_mae', 'cat_rmse', 'cat_mae'})
        self.assertEqual(next(c for c in grid if c['id'] == 'xgb_current')['params']['max_depth'], 6)


if __name__ == '__main__':
    unittest.main()
