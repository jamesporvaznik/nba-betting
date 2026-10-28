import unittest

from nba.audit_gamelogs import AuditError
from nba.aggregate_playtypes import (
    COLUMNS, PLAY_TYPES, aggregate_group, plan_repair, season_percentiles,
    select_team_rows,
)


def source(player=1, team=10, play="Isolation", possessions=10, points=10, season="2025-26"):
    return {"SEASON_ID": int("2" + season[:4]), "PLAYER_ID": player,
            "PLAYER_NAME": f"Player {player}", "TEAM_ID": team,
            "TEAM_ABBREVIATION": "BOS" if team == 10 else "NYK",
            "PLAY_TYPE": play, "TYPE_GROUPING": "offensive", "GP": 10,
            "POSS": possessions, "PTS": points, "FGM": 4, "FGA": 8,
            "FG_PCT": .5, "EFG_PCT": .5, "PPP": points / possessions,
            "PERCENTILE": .5}


class AggregationTests(unittest.TestCase):
    def test_sums_counts_and_calculates_efficiency_from_totals(self):
        a = source(possessions=100, points=120)
        b = {**source(team=20, possessions=50, points=65), "FGM": 6, "FGA": 12,
             "FG_PCT": .5, "EFG_PCT": .75, "GP": 5}
        values, merged, _ = aggregate_group([a, b])
        self.assertTrue(merged)
        self.assertEqual(values["possessions"], 150)
        self.assertEqual(values["points"], 185)
        self.assertEqual(values["points_per_possession"], 1.23)
        self.assertEqual(values["games_played"], 15)
        self.assertEqual(values["fg_pct"], .5)
        self.assertEqual(values["efg_pct"], .65)

    def test_does_not_double_count_an_existing_whole_season_total(self):
        a, b = source(), source(team=20)
        total = {**source(team=0, possessions=20, points=20), "TEAM_ABBREVIATION": "TOT"}
        self.assertEqual(select_team_rows([a, b, total]), [total])
        values, merged, _ = aggregate_group([a, b, total])
        self.assertFalse(merged)
        self.assertEqual(values["possessions"], 20)

    def test_identical_duplicate_team_rows_are_not_counted_twice(self):
        row = source()
        self.assertEqual(select_team_rows([row, dict(row)]), [row])

    def test_conflicting_duplicate_team_rows_are_rejected(self):
        with self.assertRaises(AuditError):
            select_team_rows([source(), source(points=11)])

    def test_fractional_per_game_counts_are_rejected(self):
        with self.assertRaises(AuditError):
            aggregate_group([source(possessions=1.5)])

    def test_percentile_is_ppp_not_possession_volume(self):
        groups = {(1, "2025-26", "Isolation"): [source(player=1, possessions=100, points=80)],
                  (2, "2025-26", "Isolation"): [source(player=2, possessions=10, points=12)]}
        ranks = season_percentiles(groups)
        self.assertEqual(ranks[(1, "2025-26", "Isolation")], 0)
        self.assertEqual(ranks[(2, "2025-26", "Isolation")], 1)

    def test_percentile_uses_unrounded_combined_ppp_and_average_ties(self):
        groups = {(pid, "2025-26", "Isolation"): [source(player=pid, possessions=1000, points=pts)]
                  for pid, pts in [(1, 800), (2, 1000), (3, 1000), (4, 1200)]}
        ranks = season_percentiles(groups)
        self.assertEqual([ranks[(p, "2025-26", "Isolation")] for p in [1, 2, 3, 4]], [0, .5, .5, 1])
        close = {(pid, "2025-26", "Isolation"): [source(player=pid, possessions=1000, points=pts)]
                 for pid, pts in [(1, 1001), (2, 1002)]}
        self.assertEqual(season_percentiles(close)[(1, "2025-26", "Isolation")], 0)

    def test_singleton_percentile_and_source_qualification(self):
        groups = {(1, "2025-26", "Isolation"): [source()],
                  (2, "2025-26", "Isolation"): [source(player=2, possessions=9, points=20)]}
        ranks = season_percentiles(groups)
        self.assertEqual(ranks[(1, "2025-26", "Isolation")], .5)
        self.assertIsNone(ranks[(2, "2025-26", "Isolation")])

    def test_cohorts_do_not_mix_seasons_or_play_types(self):
        groups = {(1, "2025-26", "Isolation"): [source()],
                  (2, "2024-25", "Isolation"): [source(player=2, season="2024-25", points=100)],
                  (3, "2025-26", "Transition"): [source(player=3, play="Transition", points=100)]}
        self.assertEqual(set(season_percentiles(groups).values()), {.5})

    def fixture(self):
        rows, stored = [], []
        for play in PLAY_TYPES:
            for pid in [1, 2]:
                r = source(player=pid, play=play)
                rows.append(r)
                values, _, _ = aggregate_group([r])
                stored.append({"player_id": pid, "season": "2025-26", "play_type": play,
                               **values, "poss_pct": .0909})
        rows.append({**source(team=20, possessions=20, points=30), "FGM": 12,
                     "FGA": 20, "FG_PCT": .6, "EFG_PCT": .65, "GP": 5})
        return {"2025-26": rows}, stored

    def test_plan_normalizes_after_aggregation_and_reranks_other_players(self):
        rows, stored = self.fixture()
        plan = plan_repair(rows, stored)
        after = {(r["player_id"], r["play_type"]): r for r in plan["after"]}
        self.assertEqual(after[(1, "Isolation")]["possessions"], 30)
        self.assertEqual(after[(1, "Isolation")]["poss_pct"], .2308)
        self.assertEqual(after[(1, "Transition")]["poss_pct"], .0769)
        self.assertEqual(after[(1, "Isolation")]["percentile"], 1)
        self.assertEqual(after[(2, "Isolation")]["percentile"], 0)
        self.assertEqual(after[(2, "Isolation")]["possessions"], 10)
        self.assertEqual(plan["summaries"][0]["merged_play_type_rows"], 1)
        self.assertTrue(all(set(r) == set(COLUMNS) for r in plan["after"]))

    def test_missing_keys_abort_instead_of_silently_omitting_rows(self):
        rows, stored = self.fixture()
        with self.assertRaises(AuditError):
            plan_repair(rows, stored[:-1])

    def test_incomplete_play_type_source_aborts(self):
        rows, stored = self.fixture()
        rows["2025-26"] = [r for r in rows["2025-26"] if r["PLAY_TYPE"] != "Misc"]
        with self.assertRaises(AuditError):
            plan_repair(rows, stored)


if __name__ == "__main__":
    unittest.main()
