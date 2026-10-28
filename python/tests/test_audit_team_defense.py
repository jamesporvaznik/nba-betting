import unittest
from decimal import Decimal

from nba.audit_team_defense import compare, OPPONENT_FIELDS, TEAM_FIELDS


class TeamDefenseAuditTests(unittest.TestCase):
    def fixture(self):
        base = {"GAME_ID": "0022500001", "GAME_DATE": "2025-10-21T00:00:00", "MIN": 53,
                "FGM": 40, "FGA": 90, "FG_PCT": .444, "FG3M": 10, "FG3A": 30,
                "FG3_PCT": .333, "FTM": 10, "FTA": 20, "FT_PCT": .5,
                "OREB": 10, "REB": 45, "AST": 25, "STL": 8, "BLK": 4,
                "TOV": 11, "PF": 15, "PTS": 100}
        a = dict(base, TEAM_ABBREVIATION="BOS", MATCHUP="BOS vs. NYK")
        b = dict(base, TEAM_ABBREVIATION="NYK", MATCHUP="NYK @ BOS")
        stored = []
        for own, opp, home in ((a, b, 1), (b, a, 0)):
            row = {field: opp[src] for field, src in OPPONENT_FIELDS.items()}
            row.update({field: own[src] for field, src in TEAM_FIELDS.items()})
            row.update(date="2025-10-21", team=own["TEAM_ABBREVIATION"],
                       opponent_abbrev=opp["TEAM_ABBREVIATION"], home=home,
                       game_id=f'2025-10-21_{own["TEAM_ABBREVIATION"]}_{opp["TEAM_ABBREVIATION"]}',
                       pace=90.4, team_pace=90.4, opponent_pace=90.4,
                       off_rating=100.2, def_rating=100.2,
                       opponent_avg_team_pace=90.4, opponent_pace_deviation=0,
                       opponent_avg_game_pace=90.4, opponent_game_pace_deviation=0)
            stored.append(row)
        return [a, b], stored

    def test_overtime_is_normalized_to_48_minutes(self):
        source, stored = self.fixture()
        result = compare("2025-26", source, stored, set())
        self.assertEqual(result["mismatch_counts"], {})
        stored[0]["pace"] = Decimal("99.8")
        result = compare("2025-26", source, stored, set())
        self.assertEqual(result["mismatch_counts"], {"pace": 1})

    def test_missing_extra_and_duplicate_games_are_distinguished(self):
        source, stored = self.fixture()
        extra = dict(stored[0], date="2025-12-16")
        result = compare("2025-26", source, [stored[0], stored[0], extra], set())
        self.assertEqual(len(result["missing_games"]), 1)
        self.assertEqual(len(result["extra_games"]), 1)
        self.assertEqual(result["duplicate_groups"][0]["count"], 2)

    def test_counts_and_home_flag_are_checked(self):
        source, stored = self.fixture()
        stored[0]["opponent_pts"] = 101
        stored[0]["home"] = 0
        result = compare("2025-26", source, stored, set())
        self.assertEqual(result["mismatch_counts"], {"opponent_pts": 1, "home": 1})

    def test_incomplete_source_cannot_pass(self):
        source, stored = self.fixture()
        with self.assertRaises(ValueError):
            compare("2025-26", source[:1], stored, set())


if __name__ == "__main__":
    unittest.main()
