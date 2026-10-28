import unittest

from models.apply_clusters import WEIGHTS, expected_row, verify_rows


class ApplyClusterTests(unittest.TestCase):
    def preview(self):
        return {"player_id": "123", "season": "2024-25", "primary_archetype": "0",
                "archetype_0_weight": ".80005", "archetype_1_weight": ".19995"}

    def scales(self):
        return {c: 4 if i < 7 else None for i, c in enumerate(WEIGHTS)}

    def test_rounding_matches_database_and_unused_weights_cleared(self):
        row = expected_row(self.preview(), "2024-25", 2, self.scales())
        self.assertEqual(row["archetype_0_weight"], "0.8001")
        self.assertEqual(row["archetype_1_weight"], "0.2000")
        self.assertTrue(all(row[c] is None for c in WEIGHTS[2:]))

    def test_invalid_weights_or_primary_are_rejected(self):
        for changes in ({"primary_archetype": "1"}, {"archetype_0_weight": "NaN"},
                        {"archetype_0_weight": "0.9"}, {"season": "2023-24"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                expected_row({**self.preview(), **changes}, "2024-25", 2, self.scales())

    def test_verification_preserves_old_rows_and_checks_insert_coverage(self):
        old = {**expected_row(self.preview(), "2024-25", 2, self.scales()), "created_at": "old", "updated_at": "old"}
        new = {**old, "player_id": 124, "created_at": "new"}
        plan = {"before": [old], "after": [old, new]}
        verify_rows([old, new], plan)
        for actual in ([old], [old, {**new, "archetype_8_weight": "0.1"}],
                       [{**old, "created_at": "changed"}, new]):
            with self.assertRaises(ValueError):
                verify_rows(actual, plan)


if __name__ == "__main__":
    unittest.main()
