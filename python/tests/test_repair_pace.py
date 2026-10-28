from decimal import Decimal
import unittest

from nba.repair_pace import FIELDS, build_plan, check_after, compute_pregame


SCALES = {field: 1 for field in FIELDS}


def game(day, a, b, first_id, season="2025-26", tag="regular"):
    pace = str(((Decimal(str(a)) + Decimal(str(b))) / 2).quantize(Decimal(".1")))
    rows = []
    for offset, team, opponent, own, opposing in ((0, "A", "B", a, b), (1, "B", "A", b, a)):
        row = {"id": first_id+offset, "season": season, "date": day, "team": team,
               "opponent_abbrev": opponent, "team_pace": str(own), "opponent_pace": str(opposing),
               "pace": pace, "game_id": tag, "team_pts": 100, "created_at": "old", "updated_at": "old"}
        row.update({field: "99.0" for field in FIELDS})
        rows.append(row)
    return rows


class PriorPaceRepairTests(unittest.TestCase):
    def fixture(self):
        return game("2025-10-21", 90, 110, 1) + game("2025-10-23", 100, 120, 3) + game("2025-10-25", 80, 100, 5)

    def test_first_games_have_null_baselines_and_deviations(self):
        result = compute_pregame(self.fixture(), SCALES)
        self.assertTrue(all(result[i][field] is None for i in (0, 1) for field in FIELDS))

    def test_opponent_own_pace_is_used_instead_of_its_adversaries(self):
        result = compute_pregame(self.fixture(), SCALES)
        self.assertEqual(result[2]["opponent_avg_team_pace"], "110.0")
        self.assertEqual(result[3]["opponent_avg_team_pace"], "90.0")
        self.assertEqual(result[4]["opponent_avg_team_pace"], "115.0")
        self.assertEqual(result[4]["opponent_avg_game_pace"], "105.0")

    def test_deviations_equal_current_value_minus_stored_baseline(self):
        result = compute_pregame(self.fixture(), SCALES)
        self.assertEqual(result[2]["opponent_pace_deviation"], "10.0")
        self.assertEqual(result[4]["opponent_pace_deviation"], "-15.0")
        self.assertEqual(result[4]["opponent_game_pace_deviation"], "-15.0")
        for row in result[2:]:
            self.assertEqual(Decimal(row["opponent_pace_deviation"]), Decimal(row["opponent_pace"])-Decimal(row["opponent_avg_team_pace"]))

    def test_future_game_changes_cannot_alter_earlier_results(self):
        before = self.fixture()
        baseline = compute_pregame(before, SCALES)
        changed = before[:4] + game("2025-10-25", 200, 220, 5)
        self.assertEqual(compute_pregame(changed, SCALES)[:4], baseline[:4])

    def test_same_day_rows_do_not_see_each_other_and_input_order_does_not_matter(self):
        rows = self.fixture()
        self.assertEqual(compute_pregame(list(reversed(rows)), SCALES), compute_pregame(rows, SCALES))
        self.assertTrue(all(compute_pregame(rows, SCALES)[i]["opponent_avg_team_pace"] is None for i in (0, 1)))

    def test_history_resets_between_seasons(self):
        rows = self.fixture() + game("2026-10-21", 120, 130, 7, season="2026-27")
        result = compute_pregame(rows, SCALES)
        self.assertTrue(all(row[field] is None for row in result[-2:] for field in FIELDS))

    def test_rounding_happens_before_manual_deviation_subtraction(self):
        rows = game("2025-10-21", 95, 100, 1) + game("2025-10-23", 95, 100.1, 3) + game("2025-10-25", 95, 100, 5)
        result = compute_pregame(rows, SCALES)
        self.assertEqual(result[4]["opponent_avg_team_pace"], "100.1")
        self.assertEqual(result[4]["opponent_pace_deviation"], "-0.1")

    def test_existing_cup_game_is_included_in_history(self):
        rows = self.fixture()
        rows[2]["game_id"] = rows[3]["game_id"] = "Cup Final"
        self.assertEqual(compute_pregame(rows, SCALES)[4]["opponent_avg_team_pace"], "115.0")

    def test_missing_reciprocal_game_or_missing_pace_cannot_silently_pass(self):
        with self.assertRaises(ValueError):
            compute_pregame(self.fixture()[:-1], SCALES)
        rows = self.fixture()
        rows[0]["team_pace"] = None
        with self.assertRaises(ValueError):
            compute_pregame(rows, SCALES)

    def test_duplicate_game_is_rejected(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):
            compute_pregame(rows + [dict(rows[0], id=99)], SCALES)

    def test_verification_protects_original_pace_box_scores_and_timestamps(self):
        plan = build_plan(self.fixture(), SCALES)
        result = [dict(r, updated_at="new") for r in plan["after"]]
        check_after(result, plan)
        for field, value in (("team_pace", "999"), ("team_pts", 999), ("created_at", "changed")):
            broken = [dict(r) for r in result]
            broken[0][field] = value
            with self.assertRaises(ValueError):
                check_after(broken, plan)
        repaired = build_plan(plan["after"], SCALES)
        self.assertEqual(repaired["changes"], [])
        modified = [dict(r, updated_at="changed") for r in repaired["after"]]
        with self.assertRaises(ValueError):
            check_after(modified, repaired)


if __name__ == "__main__":
    unittest.main()
