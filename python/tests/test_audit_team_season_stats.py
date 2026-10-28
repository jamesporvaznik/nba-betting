import unittest
from unittest.mock import patch
from decimal import Decimal

from nba.audit_shot_locations import ZONES
from nba.audit_team_season_stats import compare_plays, compare_shots


class TeamSeasonAuditTests(unittest.TestCase):
    def play_fixture(self):
        total = {"SEASON_ID": "22025", "TEAM_ID": 1, "TEAM_ABBREVIATION": "BOS",
                 "PLAY_TYPE": "Isolation", "TYPE_GROUPING": "Defensive", "GP": 82,
                 "POSS": 1245, "PTS": 1141, "FGM": 400, "FGA": 900,
                 "POSS_PCT": 1, "PPP": .916, "PERCENTILE": .5, "FG_PCT": .444, "EFG_PCT": .5}
        per = dict(total, POSS=15.2, PTS=13.9)
        stored = {"team_id": 1, "play_type": "Isolation", "team_abbreviation": "BOS",
                  "possessions_defended": "15.20", "points_allowed": "13.90", "poss_pct": "1.0000",
                  "percentile": ".5000", "games_played": 82, "points_allowed_per_possession": ".91",
                  "fg_pct_allowed": ".4440", "efg_pct_allowed": ".5000"}
        scales = {f: 4 for f in ("poss_pct", "percentile", "fg_pct_allowed", "efg_pct_allowed")}
        scales.update({f: 2 for f in ("possessions_defended", "points_allowed", "points_allowed_per_possession")})
        return per, total, stored, scales

    def test_importer_match_does_not_hide_ppp_rounding_loss(self):
        per, total, stored, scales = self.play_fixture()
        with patch("nba.audit_team_season_stats.PLAY_TYPES", ("Isolation",)):
            result = compare_plays("2025-26", [per], [total], [stored], scales, {1: "BOS"}, {1: {"gp": 82}})
        self.assertEqual(result["source_replay_mismatches"], [])
        self.assertEqual(result["stored_ppp_different_from_rounded_exact_totals"][0]["expected_at_storage_precision"], Decimal(".92"))

    def test_offensive_source_cannot_pass_a_defensive_audit(self):
        per, total, stored, scales = self.play_fixture()
        total["TYPE_GROUPING"] = "Offensive"
        with patch("nba.audit_team_season_stats.PLAY_TYPES", ("Isolation",)), self.assertRaises(ValueError):
            compare_plays("2025-26", [per], [total], [stored], scales, {1: "BOS"}, {1: {"gp": 82}})

    def shot_fixture(self):
        total, per, stored = {"TEAM_ID": 1}, {"TEAM_ID": 1}, []
        for zone in ZONES + ("Backcourt",):
            for stat, value in (("FGM", 5), ("FGA", 10), ("FG_PCT", .5)):
                total[f"{zone}_OPP_{stat}"] = value if zone != "Backcourt" else 0
                per[f"{zone}_OPP_{stat}"] = (.1 if stat != "FG_PCT" else .5) if zone != "Backcourt" else 0
            if zone != "Backcourt":
                stored.append({"team_id": 1, "shot_zone": zone, "fgm_allowed": ".10",
                               "fga_allowed": ".10", "fg_pct_allowed": ".5000", "frequency_pct": ".1667"})
        scales = {"fgm_allowed": 2, "fga_allowed": 2, "fg_pct_allowed": 4, "frequency_pct": 4}
        box = {1: {"gp": 82, "opponent_fgm": 30, "opponent_fga": 60}}
        return per, total, stored, scales, box

    def test_shot_coverage_and_box_totals_match(self):
        per, total, stored, scales, box = self.shot_fixture()
        result = compare_shots("2025-26", [per], [total], stored, scales, {1: "BOS"}, box)
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["source_replay_mismatches"], [])
        self.assertEqual(result["season_zone_totals_vs_regular_team_games"], [])

    def test_missing_zone_is_reported(self):
        per, total, stored, scales, box = self.shot_fixture()
        result = compare_shots("2025-26", [per], [total], stored[:-1], scales, {1: "BOS"}, box)
        self.assertEqual(len(result["missing_keys"]), 1)

    def test_per_game_match_does_not_hide_frequency_or_total_disagreement(self):
        per, total, stored, scales, box = self.shot_fixture()
        total["Restricted Area_OPP_FGA"] = 20
        result = compare_shots("2025-26", [per], [total], stored, scales, {1: "BOS"}, box)
        self.assertEqual(result["source_replay_mismatches"], [])
        self.assertEqual(result["frequency_differences_over_half_percentage_point"], 6)
        self.assertEqual(len(result["season_zone_totals_vs_regular_team_games"]), 1)


if __name__ == "__main__":
    unittest.main()
