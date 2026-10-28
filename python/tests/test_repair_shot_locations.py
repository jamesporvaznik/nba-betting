import unittest

from nba.audit_shot_locations import ZONES
from nba.repair_shot_locations import check_after, plan_repair, recover_counts


class ShotLocationRepairTests(unittest.TestCase):
    def fixture(self):
        row = {"PLAYER_ID": 1, "PLAYER_NAME": "Test Player"}
        for zone in ZONES + ("Backcourt",):
            row[f"{zone}_FGM"] = 0
            row[f"{zone}_FGA"] = 0
        row["Restricted Area_FGM"] = 10
        row["Restricted Area_FGA"] = 40
        logs = {"playerid": 1, "season": "2025-26", "gp": 82, "fgm": 10, "fga": 41}
        row["Mid-Range_FGA"] = 1
        old = {"player_id": 1, "season": "2025-26", "shot_zone": "Restricted Area",
               "fgm": ".2", "fga": ".8", "fg_pct": ".25", "frequency_pct": "1",
               "created_at": "2026-01-01", "updated_at": "2026-01-01"}
        return row, logs, old

    def test_restores_rare_zone_and_calculates_frequency_from_exact_counts(self):
        row, logs, old = self.fixture()
        plan = plan_repair({"2025-26": [row]}, [old], [logs])
        zones = {r["shot_zone"]: r for r in plan["after"]}
        self.assertEqual(zones["Mid-Range"]["fga"], "0.01")
        self.assertEqual(zones["Mid-Range"]["frequency_pct"], "0.0244")
        self.assertEqual(zones["Restricted Area"]["frequency_pct"], "0.9756")
        self.assertEqual(zones["Restricted Area"]["fga"], "0.49")
        self.assertEqual(zones["Restricted Area"]["fgm"], "0.12")
        self.assertEqual(plan["summaries"][0]["inserted_zone_rows"], 1)

    def test_nulls_are_recovered_only_when_residual_is_uniquely_determined(self):
        row, logs, _ = self.fixture()
        row["Left Corner 3_FGA"] = row["Right Corner 3_FGA"] = None
        recovered, fields = recover_counts(row, logs)
        self.assertEqual(recovered["Left Corner 3_FGA"], 0)
        self.assertEqual(len(fields), 2)
        logs["fga"] = 42
        with self.assertRaises(ValueError):
            recover_counts(row, logs)

    def test_one_unknown_zone_can_be_recovered_from_total(self):
        row, logs, _ = self.fixture()
        row["Mid-Range_FGA"] = None
        recovered, _ = recover_counts(row, logs)
        self.assertEqual(recovered["Mid-Range_FGA"], 1)

    def test_mismatched_game_totals_abort(self):
        row, logs, old = self.fixture()
        logs["fga"] = 42
        with self.assertRaises(ValueError):
            plan_repair({"2025-26": [row]}, [old], [logs])

    def test_existing_rows_cannot_be_dropped(self):
        row, logs, old = self.fixture()
        old["shot_zone"] = "Left Corner 3"
        with self.assertRaises(ValueError):
            plan_repair({"2025-26": [row]}, [old], [logs])

    def test_backcourt_does_not_change_six_zone_frequency_denominator(self):
        row, logs, old = self.fixture()
        row["Backcourt_FGA"] = 2
        logs["fga"] = 43
        plan = plan_repair({"2025-26": [row]}, [old], [logs])
        self.assertEqual(next(r for r in plan["after"] if r["shot_zone"] == "Restricted Area")["frequency_pct"], "0.9756")

    def test_result_check_detects_changed_value_or_creation_timestamp(self):
        row, logs, old = self.fixture()
        plan = plan_repair({"2025-26": [row]}, [old], [logs])
        actual = [dict(r, created_at="2026-01-01") for r in plan["after"]]
        check_after(actual, plan)
        actual[0]["fga"] = "999"
        with self.assertRaises(ValueError):
            check_after(actual, plan)
        actual = [dict(r, created_at="changed") for r in plan["after"]]
        with self.assertRaises(ValueError):
            check_after(actual, plan)


if __name__ == "__main__":
    unittest.main()
