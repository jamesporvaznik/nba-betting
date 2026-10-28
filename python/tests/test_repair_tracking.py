import unittest
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile

from nba.repair_tracking import build_plan, check_after, read_sources


class TrackingRepairTests(unittest.TestCase):
    def fixture(self):
        measures = {"Possessions": {"columns": ["TOUCHES", "AVG_SEC_PER_TOUCH"],
                    "rename": {"TOUCHES": "touches", "AVG_SEC_PER_TOUCH": "avg_sec_per_touch"}}}
        source = {"2025-26": {"Possessions": {
            "PerGame": [{"PLAYER_ID": 1, "GP": 10, "TOUCHES": 20.1, "AVG_SEC_PER_TOUCH": 2.15}],
            "Totals": [{"PLAYER_ID": 1, "GP": 10, "TOUCHES": 201, "AVG_SEC_PER_TOUCH": 2.15}]}}}
        before = [{"player_id": 1, "season": "2025-26", "touches": "19.00",
                   "avg_sec_per_touch": "2.15", "created_at": "old", "updated_at": "old",
                   "extra_metadata": "preserve"}]
        logs = [{"playerid": 1, "season": "2025-26", "gp": 10}]
        scales = {"touches": 2, "avg_sec_per_touch": 2}
        return source, before, logs, scales, measures

    def test_updates_only_different_values_and_preserves_other_fields(self):
        source, before, logs, scales, measures = self.fixture()
        plan = build_plan(source, before, logs, scales, measures)
        self.assertEqual(plan["after"][0]["touches"], "20.10")
        self.assertEqual(plan["after"][0]["created_at"], "old")
        self.assertEqual(plan["after"][0]["extra_metadata"], "preserve")
        self.assertEqual(set(plan["changes"][0]["changes"]), {"touches"})
        self.assertEqual(before[0]["touches"], "19.00")

    def test_source_nulls_are_preserved_without_imputing_zero(self):
        source, before, logs, scales, measures = self.fixture()
        for mode in ("PerGame", "Totals"):
            source["2025-26"]["Possessions"][mode][0]["AVG_SEC_PER_TOUCH"] = None
        before[0]["avg_sec_per_touch"] = None
        plan = build_plan(source, before, logs, scales, measures)
        self.assertIsNone(plan["after"][0]["avg_sec_per_touch"])

    def test_source_mode_disagreement_aborts(self):
        source, before, logs, scales, measures = self.fixture()
        source["2025-26"]["Possessions"]["Totals"][0]["AVG_SEC_PER_TOUCH"] = 4.63
        with self.assertRaises(ValueError):
            build_plan(source, before, logs, scales, measures)

    def test_duplicate_source_players_abort_instead_of_dropping_a_team(self):
        source, before, logs, scales, measures = self.fixture()
        rows = source["2025-26"]["Possessions"]["PerGame"]
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            build_plan(source, before, logs, scales, measures)

    def test_tracking_coverage_may_be_lower_than_box_score_coverage(self):
        source, before, logs, scales, measures = self.fixture()
        logs[0]["gp"] = 11
        plan = build_plan(source, before, logs, scales, measures)
        self.assertEqual(plan["after"][0]["touches"], "20.10")
        logs[0]["gp"] = 9
        with self.assertRaises(ValueError):
            build_plan(source, before, logs, scales, measures)

    def test_missing_player_coverage_aborts(self):
        source, before, logs, scales, measures = self.fixture()
        before.append(dict(before[0], player_id=2))
        with self.assertRaises(ValueError):
            build_plan(source, before, logs, scales, measures)

    def test_unchanged_rows_keep_updated_timestamp(self):
        source, before, logs, scales, measures = self.fixture()
        before[0]["touches"] = "20.10"
        plan = build_plan(source, before, logs, scales, measures)
        self.assertEqual(plan["changes"], [])
        actual = [dict(plan["after"][0], updated_at="changed")]
        with self.assertRaises(ValueError):
            check_after(actual, plan)

    def test_result_check_detects_value_creation_timestamp_and_row_loss(self):
        plan = build_plan(*self.fixture())
        actual = [dict(plan["after"][0], updated_at="new")]
        check_after(actual, plan)
        for field, value in (("created_at", "changed"), ("touches", "999"), ("extra_metadata", "changed")):
            with self.assertRaises(ValueError):
                check_after([dict(actual[0], **{field: value})], plan)
        with self.assertRaises(ValueError):
            check_after([], plan)

    def write_sources(self, directory, team=0):
        for mode in ("PerGame", "Totals"):
            params = {"Season": "2025-26", "SeasonType": "Regular Season", "PerMode": mode,
                      "PtMeasureType": "Possessions", "TeamID": team, "PlayerOrTeam": "Player",
                      "LeagueID": "00", "LastNGames": 0, "Month": 0, "OpponentTeamID": 0}
            raw = json.dumps({"parameters": params, "resultSets": [{"headers": ["PLAYER_ID", "GP", "TOUCHES", "AVG_SEC_PER_TOUCH"], "rowSet": [[1, 10, 201, 2.15]]}]})
            name = f"2025-26_Possessions_{mode}"
            (directory / f"{name}.json").write_text(raw)
            (directory / f"{name}_request.json").write_text(json.dumps({"parameters": params, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}))

    def test_changed_raw_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nba.repair_tracking.SEASONS", ("2025-26",)), patch("nba.repair_tracking.load_measures", return_value=self.fixture()[-1]):
            directory = Path(tmp)
            self.write_sources(directory)
            _, manifest = read_sources(directory)
            self.assertEqual(len(manifest), 2)
            path = directory / "2025-26_Possessions_Totals.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_sources(directory)

    def test_blank_team_filter_is_rejected_even_with_valid_checksum(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nba.repair_tracking.SEASONS", ("2025-26",)), patch("nba.repair_tracking.load_measures", return_value=self.fixture()[-1]):
            directory = Path(tmp)
            self.write_sources(directory, team="")
            with self.assertRaisesRegex(ValueError, "filters"):
                read_sources(directory)

    def test_null_shooting_efficiency_requires_zero_attempts(self):
        measures = {"CatchShoot": {"columns": ["CATCH_SHOOT_EFG_PCT"],
                    "rename": {"CATCH_SHOOT_EFG_PCT": "catch_shoot_efg_pct"}}}
        rows = [{"PLAYER_ID": 1, "GP": 10, "CATCH_SHOOT_EFG_PCT": None, "CATCH_SHOOT_FGA": 0}]
        sources = {"2025-26": {"CatchShoot": {"PerGame": rows, "Totals": rows}}}
        before = [{"player_id": 1, "season": "2025-26", "catch_shoot_efg_pct": "0.0000"}]
        logs = [{"playerid": 1, "season": "2025-26", "gp": 10}]
        plan = build_plan(sources, before, logs, {"catch_shoot_efg_pct": 4}, measures)
        self.assertIsNone(plan["after"][0]["catch_shoot_efg_pct"])
        rows[0]["CATCH_SHOOT_FGA"] = 1
        with self.assertRaises(ValueError):
            build_plan(sources, before, logs, {"catch_shoot_efg_pct": 4}, measures)


if __name__ == "__main__":
    unittest.main()
