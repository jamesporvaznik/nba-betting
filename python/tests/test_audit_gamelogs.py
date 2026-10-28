import unittest
from types import SimpleNamespace

from nba.audit_gamelogs import AuditError, compare_appearances, read_all, season_argument


def nba_row(player=1, date="2024-10-22", team="BOS", opponent="NYK"):
    return {"PLAYER_ID": player, "PLAYER_NAME": f"Player {player}",
            "GAME_DATE": date, "TEAM_ABBREVIATION": team,
            "MATCHUP": f"{team} vs. {opponent}", "GAME_ID": "0022400001"}


def db_row(player=1, date="2024-10-22", team="BOS", opponent="NYK", row_id=1):
    return {"gamelogid": row_id, "playerid": player, "date": date,
            "team": team, "opponent": opponent,
            "gameid": f"{date}_{team}_{opponent}"}


class ComparisonTests(unittest.TestCase):
    def compare(self, source, stored, players=None):
        if players is None:
            players = [{"nba_api_id": 1, "display_name": "Player 1"}]
        return compare_appearances("2024-25", source, stored, players)

    def test_matching_appearances_ignore_custom_game_ids_and_char_padding(self):
        report = self.compare([nba_row()], [db_row(team="bos  ", opponent="NYK ")])
        self.assertTrue(report["clean"])

    def test_missing_player_and_games_are_reported_with_official_id(self):
        report = self.compare([nba_row(), nba_row(player=2)], [])
        self.assertFalse(report["clean"])
        self.assertEqual(report["summary"]["missing_appearances"], 2)
        self.assertEqual(report["missing_player_records"],
                         [{"player_id": 2, "player_name": "Player 2"}])
        self.assertEqual(report["missing_appearances"][0]["nba_game_id"], "0022400001")
        self.assertTrue(report["missing_by_player"][0]["exists_in_players"])
        self.assertFalse(report["missing_by_player"][1]["exists_in_players"])

    def test_duplicates_do_not_mask_missing_records_when_row_counts_match(self):
        source = [nba_row(), nba_row(date="2024-10-23")]
        stored = [db_row(), db_row(row_id=2)]
        report = self.compare(source, stored)
        self.assertEqual(report["summary"]["missing_appearances"], 1)
        self.assertEqual(report["summary"]["extra_duplicate_database_rows"], 1)
        self.assertEqual(len(report["duplicate_database_appearances"][0]["database_records"]), 2)

    def test_unexpected_appearance_is_reported(self):
        report = self.compare([nba_row()], [db_row(), db_row(date="2024-10-23", row_id=2)])
        self.assertEqual(report["summary"]["unexpected_appearances"], 1)

    def test_source_duplicates_are_not_a_clean_audit(self):
        report = self.compare([nba_row(), nba_row()], [db_row()])
        self.assertFalse(report["clean"])
        self.assertEqual(report["summary"]["duplicate_nba_groups"], 1)

    def test_orphaned_game_log_is_detected_even_without_missing_appearances(self):
        report = self.compare([nba_row()], [db_row()], players=[])
        self.assertEqual(report["summary"]["missing_appearances"], 0)
        self.assertEqual(report["summary"]["missing_player_records"], 1)
        self.assertFalse(report["clean"])

    def test_empty_source_is_an_error_instead_of_pass(self):
        with self.assertRaises(AuditError):
            self.compare([], [])

    def test_bad_matchup_is_an_error(self):
        row = nba_row()
        row["MATCHUP"] = "unknown"
        with self.assertRaises(AuditError):
            self.compare([row], [])


class PaginationTests(unittest.TestCase):
    def factory(self, pages):
        calls = []

        class Query:
            def range(self, start, end):
                calls.append((start, end))
                return self

            def execute(self):
                return pages[len(calls) - 1]

        return Query, calls

    def test_reads_past_short_pages_when_server_cap_is_lower(self):
        factory, calls = self.factory([
            SimpleNamespace(data=[{"id": 1}, {"id": 2}], count=3),
            SimpleNamespace(data=[{"id": 3}], count=3),
        ])
        self.assertEqual(len(read_all(factory, "id")), 3)
        self.assertEqual([start for start, _ in calls], [0, 2])

    def test_repeated_page_is_rejected(self):
        factory, _ = self.factory([
            SimpleNamespace(data=[{"id": 1}], count=2),
            SimpleNamespace(data=[{"id": 1}], count=2),
        ])
        with self.assertRaises(AuditError):
            read_all(factory, "id")

    def test_changed_count_is_rejected(self):
        factory, _ = self.factory([
            SimpleNamespace(data=[{"id": 1}], count=2),
            SimpleNamespace(data=[{"id": 2}], count=3),
        ])
        with self.assertRaises(AuditError):
            read_all(factory, "id")

    def test_premature_empty_page_is_rejected(self):
        factory, _ = self.factory([SimpleNamespace(data=[], count=1)])
        with self.assertRaises(AuditError):
            read_all(factory, "id")

    def test_empty_database_is_allowed(self):
        factory, _ = self.factory([SimpleNamespace(data=[], count=0)])
        self.assertEqual(read_all(factory, "id"), [])


if __name__ == "__main__":
    unittest.main()
