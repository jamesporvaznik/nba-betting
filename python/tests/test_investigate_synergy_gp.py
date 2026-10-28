import unittest

from nba.investigate_synergy_gp import cutoff_check


class SynergyCutoffTests(unittest.TestCase):
    def test_common_cutoff_is_identified(self):
        games = [{"team": "A", "date": "2026-04-01"}, {"team": "B", "date": "2026-04-01"},
                 {"team": "A", "date": "2026-04-03"}, {"team": "B", "date": "2026-04-04"}]
        result = cutoff_check(games, {"A": 1, "B": 1})
        self.assertTrue(result["common_cutoff_possible"])
        self.assertEqual(result["closest_cutoff"]["absolute_team_game_count_difference"], 0)

    def test_mutually_incompatible_team_counts_rule_out_a_single_cutoff(self):
        games = [{"team": "A", "date": "2026-04-01"}, {"team": "A", "date": "2026-04-05"},
                 {"team": "B", "date": "2026-04-01"}, {"team": "B", "date": "2026-04-03"}]
        result = cutoff_check(games, {"A": 2, "B": 1})
        self.assertFalse(result["common_cutoff_possible"])
        self.assertGreater(result["closest_cutoff"]["absolute_team_game_count_difference"], 0)

    def test_full_season_has_no_upper_cutoff_bound(self):
        games = [{"team": "A", "date": "2026-04-01"}, {"team": "A", "date": "2026-04-05"}]
        result = cutoff_check(games, {"A": 2})
        self.assertTrue(result["common_cutoff_possible"])
        self.assertIsNone(result["earliest_required_excluded_date"])

    def test_gp_exceeding_verified_games_is_rejected(self):
        with self.assertRaises(ValueError):
            cutoff_check([{"team": "A", "date": "2026-04-01"}], {"A": 2})


if __name__ == "__main__":
    unittest.main()
