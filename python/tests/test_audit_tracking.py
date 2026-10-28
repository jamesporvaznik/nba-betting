import unittest
from unittest.mock import patch

import pandas as pd

from nba.audit_tracking import compare_measure, fetch


class TrackingAuditTests(unittest.TestCase):
    def fixture(self):
        config = {"columns": ["TOUCHES", "AVG_SEC_PER_TOUCH"],
                  "rename": {"TOUCHES": "touches", "AVG_SEC_PER_TOUCH": "avg_sec_per_touch"}}
        source = [{"PLAYER_ID": 1, "PLAYER_NAME": "Test Player", "TEAM_ABBREVIATION": "DAL",
                   "GP": 30, "TOUCHES": 10.1, "AVG_SEC_PER_TOUCH": 2.15}]
        totals = [dict(source[0], TOUCHES=303)]
        stored = {1: {"touches": "10.10", "avg_sec_per_touch": "2.15"}}
        logs = {1: {"gp": 30, "teams": ["DAL", "LAL"]}}
        scales = {"touches": 2, "avg_sec_per_touch": 2}
        return config, source, totals, stored, logs, scales

    def compare(self, fixture):
        return compare_measure("2025-26", "Possessions", *fixture)[0]

    def test_full_season_gp_is_checked_independently_of_single_team_label(self):
        result = self.compare(self.fixture())
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["source_gp_vs_verified_gamelog_differences"], [])
        self.assertEqual(len(result["traded_player_checks"]), 1)

    def test_first_team_gp_is_flagged(self):
        fixture = self.fixture()
        fixture[1][0]["GP"] = 15
        result = self.compare(fixture)
        self.assertEqual(len(result["source_gp_vs_verified_gamelog_differences"]), 1)

    def test_database_precision_is_used_without_comparing_unrounded_floats(self):
        fixture = self.fixture()
        fixture[1][0]["AVG_SEC_PER_TOUCH"] = 2.154
        self.assertEqual(self.compare(fixture)["mismatches"], [])
        fixture[1][0]["AVG_SEC_PER_TOUCH"] = 2.156
        self.assertEqual(self.compare(fixture)["mismatch_counts"], {"avg_sec_per_touch": 1})

    def test_missing_source_value_is_distinguished_from_zero(self):
        fixture = self.fixture()
        fixture[1][0]["TOUCHES"] = None
        fixture[2][0]["TOUCHES"] = None
        result = self.compare(fixture)
        self.assertEqual(result["source_null_counts"]["touches"], 1)
        self.assertEqual(result["mismatch_counts"], {"touches": 1})

    def test_source_duplicates_cannot_silently_pass(self):
        fixture = self.fixture()
        fixture[1].append(dict(fixture[1][0]))
        self.assertFalse(self.compare(fixture)["complete"])

    def test_explicit_all_team_filter_is_sent_to_nba(self):
        with patch("nba.audit_tracking.leaguedashptstats.LeagueDashPtStats") as endpoint:
            endpoint.return_value.get_data_frames.return_value = [pd.DataFrame([{"PLAYER_ID": 1}])]
            self.assertEqual(fetch("2024-25", "Possessions", "Totals"), [{"PLAYER_ID": 1}])
            self.assertEqual(endpoint.call_args.kwargs["team_id_nullable"], 0)
            self.assertEqual(endpoint.call_args.kwargs["season_type_all_star"], "Regular Season")

    def test_source_mode_disagreement_is_reported_separately_from_stored_match(self):
        fixture = self.fixture()
        fixture[2][0]["AVG_SEC_PER_TOUCH"] = 4.63
        result = self.compare(fixture)
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["totals_based_stored_mismatch_counts"], {"avg_sec_per_touch": 1})


if __name__ == "__main__":
    unittest.main()
