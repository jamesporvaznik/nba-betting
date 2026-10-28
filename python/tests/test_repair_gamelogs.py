import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nba.audit_gamelogs import AuditError, compare_appearances
from nba.repair_gamelogs import (
    build_stints, check_audit, database_key, fingerprint, game_record, main,
    number, player_record, render_sql,
)


def source_row(player=1, date="2025-10-22", team="BOS", opponent="NYK"):
    return {
        "PLAYER_ID": player, "PLAYER_NAME": f"Player {player}", "GAME_DATE": date,
        "TEAM_ABBREVIATION": team, "TEAM_ID": 1610612738,
        "MATCHUP": f"{team} vs. {opponent}", "GAME_ID": "0022500001", "WL": "W",
        "MIN": 20.5, "PTS": 10, "REB": 4, "AST": 2, "STL": 1, "BLK": 0,
        "TOV": 1, "PF": 2, "FGM": 4, "FGA": 8, "FG3M": 1, "FG3A": 3,
        "FTM": 1, "FTA": 2, "OREB": 1, "DREB": 3, "FG_PCT": .5,
        "FG3_PCT": .333, "FT_PCT": .5, "PLUS_MINUS": 5,
    }


class RepairTests(unittest.TestCase):
    def test_missing_weight_is_null_and_profile_does_not_assign_historical_team(self):
        profile = player_record({"PERSON_ID": 2, "DISPLAY_FIRST_LAST": "Player 2",
                                 "HEIGHT": "6-7", "WEIGHT": "", "BIRTHDATE": "",
                                 "ROSTERSTATUS": "Active"}, 2)
        self.assertIsNone(profile["weight_lbs"])
        self.assertIsNone(profile["birth_date"])
        self.assertEqual(profile["height_inches"], 79)
        self.assertTrue(profile["is_active"])
        self.assertNotIn("team_id", profile)

    def test_wrong_profile_id_is_rejected(self):
        with self.assertRaises(AuditError):
            player_record({"PERSON_ID": 3}, 2)

    def test_fractional_integer_stat_is_rejected(self):
        with self.assertRaises(AuditError):
            number(1.5, integer=True)

    def test_transform_preserves_existing_game_id_and_three_point_columns(self):
        game = game_record(source_row(), "2025-26")
        self.assertEqual(game["gameid"], "2025-10-22_BOS_NYK")
        self.assertEqual(game["3PM"], 1)
        self.assertEqual(game["minutes"], 20.5)

    def test_bad_box_scores_fail_before_sql_is_generated(self):
        for change in ({"FGM": 20}, {"REB": 100}, {"MIN": -1}, {"FG_PCT": 2}):
            with self.subTest(change=change), self.assertRaises(AuditError):
                game_record({**source_row(), **change}, "2025-26")

    def test_return_to_team_creates_a_third_stint(self):
        games = [game_record(source_row(date=f"2025-10-{day}", team=team), "2025-26")
                 for day, team in ((22, "BOS"), (24, "NYK"), (26, "BOS"))]
        stints = build_stints(list(reversed(games)), {"BOS": 10, "NYK": 20})
        self.assertEqual([s["team_id"] for s in stints], [10, 20, 10])
        self.assertEqual([s["stint_number"] for s in stints], [1, 2, 3])

    def test_gap_without_team_change_does_not_split_stint(self):
        games = [game_record(source_row(date=date), "2025-26")
                 for date in ("2025-10-22", "2026-04-12")]
        stints = build_stints(games, {"BOS": 10})
        self.assertEqual(len(stints), 1)
        self.assertEqual(stints[0]["last_game_date"], "2026-04-12")

    def test_missing_team_mapping_is_rejected(self):
        with self.assertRaises(AuditError):
            build_stints([game_record(source_row(), "2025-26")], {})

    def test_fingerprint_is_order_independent_but_detects_duplicates(self):
        a = (1, "2025-10-22", "BOS", "NYK")
        b = (2, "2025-10-22", "BOS", "NYK")
        self.assertEqual(fingerprint([a, b]), fingerprint([b, a]))
        self.assertNotEqual(fingerprint([a, b]), fingerprint([a, b, b]))

    def test_stale_audit_is_rejected(self):
        report = compare_appearances("2025-26", [source_row()], [], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"reports": [report]}))
            check_audit(path, "2025-26", report)
            changed = {**report, "summary": {**report["summary"], "database_rows": 1}}
            with self.assertRaises(AuditError):
                check_audit(path, "2025-26", changed)

    def test_sql_quotes_apostrophes_in_player_names(self):
        profile = player_record({"PERSON_ID": 2, "DISPLAY_FIRST_LAST": "N'Example"}, 2)
        sql = render_sql("2025-26", [profile], [], [], fingerprint([]), fingerprint([]), 0)
        self.assertIn("N''Example", sql)
        self.assertIn("BEGIN;", sql)
        self.assertIn("COMMIT;", sql)

    def test_preview_exports_sql_using_a_client_with_only_read_operations(self):
        stored = {**game_record(source_row(), "2025-26"), "gamelogid": 1}
        tables = {
            "players": [{"id": 1, "nba_api_id": 1, "display_name": "Player 1"}],
            "player_gamelog": [stored], "player_team_stints": [],
            "teams": [{"id": 1, "nba_api_id": 1610612738, "abbreviation": "BOS"}],
        }

        class Query:
            def __init__(self, table):
                self.rows = tables[table]

            def select(self, *args, **kwargs):
                return self

            def eq(self, *args):
                return self

            def order(self, *args):
                return self

            def range(self, start, end):
                self.page = self.rows[start:end + 1]
                return self

            def execute(self):
                return SimpleNamespace(data=self.page, count=len(self.rows))

        # No insert/update/upsert/delete/rpc methods exist on this fake client.
        client = SimpleNamespace(table=Query)
        with tempfile.TemporaryDirectory() as directory:
            sql_path = Path(directory) / "repair.sql"
            with patch("nba.repair_gamelogs.create_client", return_value=client), \
                 patch("nba.repair_gamelogs.dotenv_values", return_value={"SUPABASE_URL": "test", "SUPABASE_KEY": "test"}), \
                 patch("nba.repair_gamelogs.fetch_nba_rows", return_value=[source_row(), source_row(player=2)]), \
                 patch("nba.repair_gamelogs.fetch_profile", return_value=player_record({"PERSON_ID": 2, "DISPLAY_FIRST_LAST": "Player 2"}, 2)), \
                 patch("nba.repair_gamelogs.time.sleep"):
                result = main(["--season", "2025-26", "--sql", str(sql_path)])
            self.assertEqual(result, 0)
            self.assertTrue(sql_path.is_file())
            self.assertIn('"playerid": 2', sql_path.read_text())
            self.assertNotIn('"playerid": 1', sql_path.read_text())
            self.assertEqual(len(tables["player_gamelog"]), 1)


if __name__ == "__main__":
    unittest.main()
