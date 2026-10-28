import json
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from models.select_clusters import choose_count, evaluate_k, sampling_stability


class ClusterSelectionTests(unittest.TestCase):
    def candidate(self, k, bic, size=10, converged=True):
        return {"clusters": k, "bic": bic, "all_seed_fits_converged": converged,
                "eligible_for_comparable_players": k >= 2 and size >= 5}

    def test_count_is_selected_by_bic_without_preferring_previous_count(self):
        results = [self.candidate(2, 100), self.candidate(7, 50), self.candidate(9, 10)]
        raw, selected = choose_count(results)
        self.assertEqual(raw["clusters"], 9)
        self.assertEqual(selected["clusters"], 9)

    def test_tiny_group_winner_is_reported_separately_from_usable_winner(self):
        results = [self.candidate(2, 100), self.candidate(7, 50), self.candidate(9, 10, size=1)]
        raw, selected = choose_count(results)
        self.assertEqual(raw["clusters"], 9)
        self.assertEqual(selected["clusters"], 7)

    def test_nonconverged_candidate_cannot_win(self):
        _, selected = choose_count([self.candidate(2, 100), self.candidate(7, 1, converged=False)])
        self.assertEqual(selected["clusters"], 2)

    def test_no_usable_groups_is_explicit_error(self):
        with self.assertRaises(ValueError):
            choose_count([self.candidate(1, 100), self.candidate(2, 1, size=1)])

    def test_known_two_group_data_selects_two_and_serializes_diagnostics(self):
        rng = np.random.default_rng(17)
        matrix = np.vstack([rng.normal(-5, .25, (60, 2)), rng.normal(5, .25, (60, 2))])
        with threadpool_limits(limits=1):
            first, _ = evaluate_k(matrix, 1, [7, 42], 2, 200, 5)
            second, model = evaluate_k(matrix, 2, [7, 42], 2, 200, 5)
            stability = sampling_stability(matrix, model, 2, 200, 3)
        _, chosen = choose_count([first, second])
        self.assertEqual(chosen["clusters"], 2)
        self.assertEqual(second["smallest_cluster"], 60)
        self.assertAlmostEqual(second["seed_ari_median"], 1)
        self.assertAlmostEqual(stability["median_ari"], 1)
        self.assertIsInstance(json.dumps(second), str)


if __name__ == "__main__":
    unittest.main()
