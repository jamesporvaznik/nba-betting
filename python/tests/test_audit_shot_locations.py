import unittest

from nba.audit_shot_locations import ZONES, compare


class ShotLocationAuditTests(unittest.TestCase):
    def fixture(self):
        totals = {"PLAYER_ID": 1, "PLAYER_NAME": "Test Player"}
        averages = dict(totals)
        for zone in ZONES + ("Backcourt",):
            for field in ("FGM", "FGA", "FG_PCT"):
                totals[f"{zone}_{field}"] = 0
                averages[f"{zone}_{field}"] = 0
        totals.update({"Restricted Area_FGM": 20, "Restricted Area_FGA": 40,
                       "Restricted Area_FG_PCT": .5})
        averages.update({"Restricted Area_FGM": .5, "Restricted Area_FGA": 1,
                         "Restricted Area_FG_PCT": .5})
        stored = [{"player_id": 1, "shot_zone": "Restricted Area", "fgm": .5,
                   "fga": 1, "fg_pct": .5, "frequency_pct": 1}]
        logs = {1: {"teams": 2, "fgm": 20, "fga": 40}}
        return totals, averages, stored, logs

    def test_traded_player_whole_season_matches_both_teams(self):
        totals, averages, stored, logs = self.fixture()
        result = compare("2025-26", [totals], [averages], stored, logs)
        self.assertEqual(result["mismatch_counts"], {})
        self.assertEqual(result["zone_totals_vs_gamelog_mismatches"], [])
        self.assertEqual(result["multi_team_player_seasons_present_in_source_and_gamelogs"], 1)

    def test_rounded_away_zone_is_distinguished_from_importer_mismatch(self):
        totals, averages, stored, logs = self.fixture()
        totals["Mid-Range_FGA"] = 1
        logs[1]["fga"] = 41
        result = compare("2025-26", [totals], [averages], stored, logs)
        self.assertEqual(result["missing_per_game_rows"], [])
        self.assertEqual(len(result["missing_positive_totals_zones"]), 1)
        self.assertEqual(len(result["positive_zones_rounded_to_zero_in_per_game_source"]), 1)

    def test_first_team_only_totals_are_detected(self):
        totals, averages, stored, logs = self.fixture()
        logs[1].update(fgm=30, fga=60)
        result = compare("2025-26", [totals], [averages], stored, logs)
        self.assertEqual(len(result["zone_totals_vs_gamelog_mismatches"]), 2)

    def test_backcourt_counts_are_included_in_cross_check_but_not_zone_frequency(self):
        totals, averages, stored, logs = self.fixture()
        totals["Backcourt_FGA"] = 2
        logs[1]["fga"] = 42
        result = compare("2025-26", [totals], [averages], stored, logs)
        self.assertEqual(result["zone_totals_vs_gamelog_mismatches"], [])
        self.assertEqual(result["frequency_sums_outside_rounding"], [])
        self.assertEqual(len(result["players_with_backcourt_attempts_excluded_by_six_zone_design"]), 1)

    def test_null_zone_is_reported_as_unknown_instead_of_zero(self):
        totals, averages, stored, logs = self.fixture()
        for row in (totals, averages):
            for stat in ("FGM", "FGA", "FG_PCT"):
                row[f"Mid-Range_{stat}"] = None
        result = compare("2025-26", [totals], [averages], stored, logs)
        self.assertEqual(len(result["source_null_zone_records_not_assumed_zero"]), 1)
        self.assertEqual(len(result["incomplete_source_player_totals_not_compared"]), 2)


if __name__ == "__main__":
    unittest.main()
