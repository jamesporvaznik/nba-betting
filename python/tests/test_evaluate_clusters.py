import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from models.evaluate_clusters import geometry, sample_stability


class ClusterEvaluationTests(unittest.TestCase):
    def test_neighbor_metrics_exclude_self_and_have_chance_baseline(self):
        rng = np.random.default_rng(17)
        matrix = np.vstack([rng.normal(-5, .1, (20, 2)), rng.normal(5, .1, (20, 2))])
        labels = np.repeat([0, 1], 20)
        result = geometry(matrix, labels)
        self.assertEqual(result["nearest5_same_cluster_fraction"], 1)
        self.assertEqual(result["nearest10_excluded_by_cluster_fraction"], 0)
        self.assertAlmostEqual(result["random_pair_same_cluster_fraction"], 19 / 39)
        self.assertEqual(result["nearest5_chance_adjusted_agreement"], 1)
        self.assertEqual(result["negative_silhouette_fraction"], 0)

    def test_sample_refits_preserve_obvious_groups_after_rescaling(self):
        rng = np.random.default_rng(2)
        raw = np.vstack([rng.normal(-5, .1, (40, 2)), rng.normal(5, .1, (40, 2))])
        raw[:, 1] *= 100
        with threadpool_limits(limits=1):
            result = sample_stability(raw, np.repeat([0, 1], 40), 2, repeats=3)
        self.assertTrue(result["all_converged"])
        self.assertEqual(result["median_ari"], 1)
        self.assertEqual(result["median_original_pair_retention"], 1)


if __name__ == "__main__":
    unittest.main()
