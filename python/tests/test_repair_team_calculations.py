from decimal import Decimal
import unittest
from unittest.mock import patch

from nba.repair_team_calculations import PLAY_TABLE, SHOT_TABLE, build_plan, check_after


class TeamCalculationRepairTests(unittest.TestCase):
    def fixture(self):
        total = {"TEAM_ID": 1, "season": "2025-26", "SEASON_ID": "22025", "PLAY_TYPE": "Isolation",
                 "TYPE_GROUPING": "Defensive", "GP": 82, "POSS": 1245, "PTS": 1141}
        per = dict(total, POSS=15.2, PTS=13.9)
        total_shots = {"TEAM_ID": 1, "season": "2025-26", "Restricted Area_OPP_FGA": 10,
                       "Restricted Area_OPP_FGM": 5, "Mid-Range_OPP_FGA": 20, "Mid-Range_OPP_FGM": 10}
        per_shots = dict(total_shots, **{"Restricted Area_OPP_FGA": .1, "Restricted Area_OPP_FGM": .1,
                                        "Mid-Range_OPP_FGA": .2, "Mid-Range_OPP_FGM": .1})
        sources = {PLAY_TABLE: {"PerGame": [per], "Totals": [total]},
                   SHOT_TABLE: {"PerGame": [per_shots], "Totals": [total_shots]}}
        before = {PLAY_TABLE: [{"team_id": 1, "season": "2025-26", "play_type": "Isolation",
                               "games_played": 82, "possessions_defended": "15.20", "points_allowed": "13.90",
                               "points_allowed_per_possession": ".91", "percentile": ".5000",
                               "created_at": "old", "updated_at": "old"}], SHOT_TABLE: []}
        for zone, fga, frequency in (("Restricted Area", ".10", ".3400"), ("Mid-Range", ".20", ".6600")):
            before[SHOT_TABLE].append({"team_id": 1, "season": "2025-26", "shot_zone": zone,
                                       "fgm_allowed": ".10", "fga_allowed": fga,
                                       "fg_pct_allowed": ".5000", "frequency_pct": frequency,
                                       "created_at": "old", "updated_at": "old"})
        scales = {PLAY_TABLE: {"possessions_defended": 2, "points_allowed": 2, "points_allowed_per_possession": 2},
                  SHOT_TABLE: {"fgm_allowed": 2, "fga_allowed": 2, "frequency_pct": 4}}
        return sources, before, scales

    def plan(self, fixture=None):
        with patch("nba.repair_team_calculations.ZONES", ("Restricted Area", "Mid-Range")):
            return build_plan(*(fixture or self.fixture()))

    def test_ppp_uses_totals_instead_of_dividing_rounded_per_game_values(self):
        plan = self.plan()
        self.assertEqual(plan["after"][PLAY_TABLE][0]["points_allowed_per_possession"], "0.92")
        self.assertEqual(plan["after"][PLAY_TABLE][0]["games_played"], 82)
        self.assertEqual(plan["after"][PLAY_TABLE][0]["possessions_defended"], "15.20")
        self.assertEqual(plan["after"][PLAY_TABLE][0]["percentile"], ".5000")

    def test_shares_use_exact_zone_totals_and_keep_counts_and_fg_pct(self):
        plan = self.plan()
        after = {r["shot_zone"]: r for r in plan["after"][SHOT_TABLE]}
        self.assertEqual(after["Restricted Area"]["frequency_pct"], "0.3333")
        self.assertEqual(after["Mid-Range"]["frequency_pct"], "0.6667")
        self.assertEqual(after["Restricted Area"]["fga_allowed"], ".10")
        self.assertEqual(after["Restricted Area"]["fg_pct_allowed"], ".5000")
        self.assertEqual(sum(Decimal(r["frequency_pct"]) for r in after.values()), 1)

    def test_unexpected_gp_difference_refuses_to_change_the_data(self):
        fixture = self.fixture()
        fixture[0][PLAY_TABLE]["Totals"][0]["GP"] = 80
        with self.assertRaises(ValueError):
            self.plan(fixture)

    def test_missing_stored_row_cannot_be_silently_inserted(self):
        fixture = self.fixture()
        fixture[1][SHOT_TABLE].pop()
        with self.assertRaises(ValueError):
            self.plan(fixture)

    def test_zero_possessions_cannot_produce_a_fabricated_ppp(self):
        fixture = self.fixture()
        fixture[0][PLAY_TABLE]["Totals"][0]["POSS"] = 0
        with self.assertRaises(ValueError):
            self.plan(fixture)

    def test_result_validation_protects_gp_and_other_metrics_and_timestamps(self):
        plan = self.plan()
        actual = {table: [dict(r, updated_at="new") for r in rows] for table, rows in plan["after"].items()}
        check_after(actual, plan)
        for column, value in (("games_played", 99), ("percentile", ".1"), ("created_at", "changed")):
            broken = {table: [dict(r) for r in rows] for table, rows in actual.items()}
            broken[PLAY_TABLE][0][column] = value
            with self.assertRaises(ValueError):
                check_after(broken, plan)

    def test_unchanged_row_keeps_update_timestamp(self):
        fixture = self.fixture()
        fixture[1][PLAY_TABLE][0]["points_allowed_per_possession"] = ".92"
        plan = self.plan(fixture)
        self.assertEqual(plan["changes"][PLAY_TABLE], [])
        actual = {table: [dict(r) for r in rows] for table, rows in plan["after"].items()}
        actual[PLAY_TABLE][0]["updated_at"] = "changed"
        with self.assertRaises(ValueError):
            check_after(actual, plan)


if __name__ == "__main__":
    unittest.main()
