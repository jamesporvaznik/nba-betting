import unittest

import numpy as np
import pandas as pd

from models.audit_clusters import SnapshotClient, compare_inputs, neighbor_checks, weight_checks


class ClusterAuditTests(unittest.TestCase):
    def test_rounded_weights_and_unused_null_components_are_valid(self):
        rows = [{"player_id": 1, "primary_archetype": 0, "archetype_0_weight": .3333,
                 "archetype_1_weight": .3333, "archetype_2_weight": .3333, "archetype_7_weight": None}]
        result = weight_checks(rows)
        self.assertEqual(result["invalid_weight_players"], [])
        self.assertEqual(result["weight_sums_outside_rounding"], [])
        self.assertEqual(result["primary_label_weight_mismatches"], [])

    def test_wrong_sum_and_primary_assignment_are_distinguished(self):
        row = {"player_id": 1, "primary_archetype": 0, "archetype_0_weight": .2,
               "archetype_1_weight": .6}
        result = weight_checks([row])
        self.assertEqual(result["weight_sums_outside_rounding"], [1])
        self.assertEqual(result["primary_label_weight_mismatches"], [1])

    def test_nonfinite_weight_is_invalid(self):
        result = weight_checks([{"player_id": 1, "primary_archetype": 0, "archetype_0_weight": np.nan}])
        self.assertEqual(result["invalid_weight_players"], [1])

    def test_snapshot_query_filters_before_paginating_and_coerces_numeric_values(self):
        rows = [{"player_id": 1, "season": "2024-25", "poss_pct": ".1"},
                {"player_id": 2, "season": "2025-26", "poss_pct": ".2"},
                {"player_id": 3, "season": "2025-26", "poss_pct": ".3"}]
        client = SnapshotClient(rows, [])
        result = client.table("player_playtype_stats").select("player_id,poss_pct").eq("season", "2025-26").range(1, 1).execute()
        self.assertEqual(result.data, [{"player_id": 3, "poss_pct": .3}])
        self.assertEqual(rows[2]["poss_pct"], ".3")

    def test_input_changes_are_compared_by_player_id_not_row_order(self):
        old = pd.DataFrame({"style": [1., 2.]}, index=[1, 2])
        new = pd.DataFrame({"style": [2., 1.5, 3.]}, index=[2, 1, 3])
        result = compare_inputs(old, new)
        self.assertEqual(result["changed_players"], [1])
        self.assertEqual(result["new_current_players"], [3])

    def test_neighbor_agreement_has_a_cluster_size_adjusted_random_baseline(self):
        features = pd.DataFrame({"style": [0., .01, .02, .03, 10., 10.01, 10.02, 10.03]}, index=list(range(8)))
        rows = [{"player_id": i, "primary_archetype": int(i >= 4)} for i in range(8)]
        result = neighbor_checks(features, rows, {})
        self.assertAlmostEqual(result["mean_same_primary_cluster_among_five_nearest_styles"], .6)
        self.assertAlmostEqual(result["random_same_cluster_baseline"], 3/7)
        self.assertEqual(result["invalid_feature_players"], [])


if __name__ == "__main__":
    unittest.main()
