import unittest

import numpy as np
import pandas as pd

from models.features.cluster_matchup import add_cluster_matchups, cluster_groups


class ClusterMatchupTests(unittest.TestCase):
    def inputs(self):
        rows = []
        for player in [1, 2, 3]:
            for day in range(13):
                rows.append({'playerid': player, 'season': '2023-24',
                             'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                             'team': f'T{player}', 'opponent': 'B',
                             'points': 10., 'rebounds': 5., 'assists': 2., 'minutes': 20.})
        clusters = pd.DataFrame([{'player_id': p, 'season': '2022-23', 'primary_archetype': c}
                                 for p, c in [(1, 0), (2, 0), (3, 1)]])
        return pd.DataFrame(rows), clusters

    def feature_cols(self, frame):
        return [c for c in frame if c.startswith(('general_', 'cluster_', 'lag_cluster'))]

    def test_hand_calculated_peer_residual_shrinkage_decay_and_minutes(self):
        games, clusters = self.inputs()
        event = games.playerid.eq(2) & games.date.eq('2024-01-11')
        games.loc[event, ['points', 'minutes']] = [20, 40]
        result = add_cluster_matchups(games, clusters, half_life_days=1, shrinkage=2)
        row = result[result.playerid.eq(1) & result.date.eq('2024-01-12')].iloc[0]
        # One peer outcome, now with weight .5: raw +10, minutes-adjusted zero.
        self.assertAlmostEqual(row.cluster_decayed_appearances, .5)
        self.assertAlmostEqual(row.cluster_points_rolling10_delta, 5 / 2.5)
        self.assertAlmostEqual(row.cluster_points_rolling10_minutes_delta, 0)
        # All-player control also includes player 3's zero residual.
        self.assertAlmostEqual(row.general_decayed_appearances, 1.)
        self.assertAlmostEqual(row.general_points_rolling10_delta, 5 / 3.)

    def test_current_and_future_results_do_not_affect_features(self):
        games, clusters = self.inputs()
        before = add_cluster_matchups(games, clusters)
        games.loc[games.date >= '2024-01-12', ['points', 'rebounds', 'assists', 'minutes']] *= 5
        after = add_cluster_matchups(games, clusters)
        mask = before.date <= '2024-01-12'
        pd.testing.assert_frame_equal(before.loc[mask, self.feature_cols(before)],
                                      after.loc[mask, self.feature_cols(after)])

    def test_five_and_ten_game_baselines_capture_different_recent_roles(self):
        games, clusters = self.inputs()
        games.loc[games.playerid.eq(2) & (games.date < '2024-01-06'), 'points'] = 0
        games.loc[games.playerid.eq(2) & games.date.between('2024-01-06', '2024-01-10'), 'points'] = 20
        result = add_cluster_matchups(games, clusters, half_life_days=1, shrinkage=2)
        row = result[result.playerid.eq(1) & result.date.eq('2024-01-12')].iloc[0]
        self.assertAlmostEqual(row.cluster_points_rolling10_delta, 0)
        self.assertAlmostEqual(row.cluster_points_rolling5_delta, -5 / 2.5)

    def test_target_history_excluded_and_rows_order_independent(self):
        games, clusters = self.inputs()
        before = add_cluster_matchups(games, clusters)
        games.loc[games.playerid.eq(1), 'points'] = np.arange(13) ** 2
        after = add_cluster_matchups(games.sample(frac=1, random_state=10), clusters)
        mask = before.playerid.eq(1)
        np.testing.assert_allclose(before.loc[mask, self.feature_cols(before)].to_numpy(float),
                                   after.loc[mask, self.feature_cols(after)].to_numpy(float))

    def test_only_previous_cluster_used_missing_retained_and_labels_not_features(self):
        games, clusters = self.inputs()
        clusters = clusters[clusters.player_id.ne(3)]
        current = clusters.assign(season='2023-24', primary_archetype=99)
        result = add_cluster_matchups(games, pd.concat([clusters, current]))
        self.assertEqual(len(result), len(games))
        self.assertTrue(result.loc[result.playerid.eq(1), 'lag_cluster'].eq(0).all())
        missing = result[result.playerid.eq(3)]
        self.assertTrue(missing.lag_cluster_available.eq(0).all())
        self.assertTrue(missing.cluster_decayed_appearances.eq(0).all())
        for columns in cluster_groups('points').values():
            self.assertNotIn('lag_cluster', columns)

    def test_new_season_and_new_opponent_have_no_history(self):
        games, clusters = self.inputs()
        games.loc[games.date.eq('2024-01-12'), 'opponent'] = 'C'
        games.loc[games.date.eq('2024-01-13'), 'season'] = '2024-25'
        result = add_cluster_matchups(games, clusters)
        self.assertTrue(result.loc[result.date >= '2024-01-12', 'general_decayed_appearances'].eq(0).all())

    def test_current_season_mode_uses_own_season_and_labels_output(self):
        games, clusters = self.inputs()
        current = clusters.assign(season='2023-24', primary_archetype=7)
        result = add_cluster_matchups(games, pd.concat([clusters, current]), profile_timing='completed_current')
        self.assertTrue(result.retrospective_cluster.eq(7).all())
        self.assertNotIn('lag_cluster', result)
        self.assertIn('retrospective_cluster_available', cluster_groups('points', 'completed_current')['cluster_rolling10'])
        with self.assertRaises(ValueError):
            add_cluster_matchups(games, clusters, profile_timing='invalid')

    def test_current_mode_still_excludes_current_day_outcomes(self):
        games, clusters = self.inputs()
        clusters = clusters.assign(season='2023-24')
        before = add_cluster_matchups(games, clusters, profile_timing='completed_current')
        games.loc[games.date >= '2024-01-12', ['points', 'rebounds', 'assists', 'minutes']] *= 5
        after = add_cluster_matchups(games, clusters, profile_timing='completed_current')
        mask = before.date <= '2024-01-12'
        columns = [c for c in before if c.startswith(('general_', 'cluster_', 'retrospective_'))]
        pd.testing.assert_frame_equal(before.loc[mask, columns], after.loc[mask, columns])

    def test_zero_current_minutes_not_filtered_and_duplicate_cluster_rejected(self):
        games, clusters = self.inputs()
        games.loc[games.playerid.eq(2) & games.date.eq('2024-01-11'),
                  ['minutes', 'points', 'rebounds', 'assists']] = 0
        result = add_cluster_matchups(games, clusters)
        row = result[result.playerid.eq(1) & result.date.eq('2024-01-12')].iloc[0]
        self.assertGreater(row.cluster_decayed_appearances, 0)
        self.assertLess(row.cluster_points_rolling10_delta, 0)
        self.assertAlmostEqual(row.cluster_points_rolling10_minutes_delta, 0)
        with self.assertRaises(ValueError):
            add_cluster_matchups(games, pd.concat([clusters, clusters.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
