import unittest
from unittest.mock import patch

from nba.audit_master_and_pace import audit_identities, audit_pace


class MasterAndPaceAuditTests(unittest.TestCase):
    def fixture(self):
        teams = [{"id": 9+i, "nba_api_id": 100+i, "abbreviation": f"T{i}",
                  "full_name": f"Team {i}", "conference": "Eastern", "division": "Atlantic"} for i in range(30)]
        standings = [{"TeamID": 100+i, "Conference": "East", "Division": "Atlantic"} for i in range(30)]
        static = [{"id": t["nba_api_id"], "abbreviation": t["abbreviation"], "full_name": t["full_name"]} for t in teams]
        players = [{"id": 1, "nba_api_id": 202, "first_name": "Jose", "display_name": "Jose Smith",
                    "team_id": 9, "is_active": True, "bball_ref_id": "smithjo01"}]
        directory = [{"PERSON_ID": 202, "DISPLAY_FIRST_LAST": "José Smith", "ROSTERSTATUS": 1, "TEAM_ID": 100}]
        return players, teams, directory, standings, static

    def test_internal_team_id_is_not_confused_with_nba_id(self):
        players, teams, directory, standings, static = self.fixture()
        refs = [("team_nba_id", "player_team_stints", "team_id", [{"value": 100, "records": 2}])]
        with patch("nba.audit_master_and_pace.static_teams.get_teams", return_value=static):
            result = audit_identities(players, teams, directory, standings, refs)
        self.assertEqual(result["players"]["invalid_internal_team_references"], [])
        self.assertEqual(result["players"]["current_roster_team_refresh_candidates"], [])
        self.assertEqual(result["reference_checks"][0]["orphaned_ids"], [])
        self.assertEqual(result["teams"]["mapping_differences"], [])

    def test_accents_are_reported_without_being_called_a_different_identity(self):
        players, teams, directory, standings, static = self.fixture()
        with patch("nba.audit_master_and_pace.static_teams.get_teams", return_value=static):
            result = audit_identities(players, teams, directory, standings, [])
        self.assertEqual(len(result["players"]["raw_name_differences"]), 1)
        self.assertEqual(result["players"]["normalized_name_differences"], [])

    def test_invalid_foreign_key_and_current_roster_change_are_distinguished(self):
        players, teams, directory, standings, static = self.fixture()
        players[0]["team_id"] = 999
        refs = [("player_nba_id", "player_gamelog", "playerid", [{"value": 999, "records": 3}])]
        with patch("nba.audit_master_and_pace.static_teams.get_teams", return_value=static):
            result = audit_identities(players, teams, directory, standings, refs)
        self.assertEqual(len(result["players"]["invalid_internal_team_references"]), 1)
        self.assertEqual(len(result["players"]["current_roster_team_refresh_candidates"]), 1)
        self.assertEqual(result["reference_checks"][0]["orphaned_ids"][0]["records"], 3)

    def test_pace_wrong_grouping_and_current_day_exclusion_are_detected(self):
        rows = []
        for day, a, b in (("2026-01-01", 90, 110), ("2026-01-03", 100, 120)):
            rows += [{"season": "2025-26", "date": day, "team": "A", "opponent_abbrev": "B",
                      "team_pace": a, "opponent_avg_team_pace": 95},
                     {"season": "2025-26", "date": day, "team": "B", "opponent_abbrev": "A",
                      "team_pace": b, "opponent_avg_team_pace": 115}]
        result = audit_pace(rows)["seasons"][0]
        self.assertEqual(result["stored_average_disagrees_with_importer_formula"], 0)
        self.assertEqual(result["stored_average_disagrees_with_opponent_own_pace"], 4)
        self.assertEqual(result["rows_without_any_prior_opponent_games"], 2)
        self.assertEqual(result["rows_with_prior_history_where_full_season_average_differs"], 2)

    def test_correct_prior_baselines_with_initial_nulls_pass(self):
        rows = []
        for day, a, b, prior_a, prior_b in (("2026-01-01", 90, 110, None, None),
                                           ("2026-01-03", 100, 120, 90, 110)):
            rows += [{"season": "2025-26", "date": day, "team": "A", "opponent_abbrev": "B",
                      "team_pace": a, "opponent_avg_team_pace": prior_b},
                     {"season": "2025-26", "date": day, "team": "B", "opponent_abbrev": "A",
                      "team_pace": b, "opponent_avg_team_pace": prior_a}]
        self.assertTrue(audit_pace(rows)["strict_prior_team_average_checks_pass"])


if __name__ == "__main__":
    unittest.main()
