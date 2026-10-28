import unittest

import numpy as np
import pandas as pd

from models.full_training_review import assess, calibration_fit, development_folds, packages, select_package


class FullTrainingReviewTests(unittest.TestCase):
    def test_development_folds_exclude_test_and_future_training_rows(self):
        frame = pd.DataFrame({'season': ['2022-23', '2023-24', '2024-25', '2025-26'],
                              'date': pd.to_datetime(['2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01'])})
        folds = list(development_folds(frame))
        self.assertEqual(len(folds), 2)
        for season, train, validation in folds:
            self.assertNotIn('2025-26', set(train.season) | set(validation.season))
            self.assertLess(train.date.max(), validation.date.min())
        self.assertEqual(set(folds[-1][1].season), {'2022-23', '2023-24'})

    def test_feature_addition_must_help_both_folds_by_minimum_amount(self):
        rows = []
        for season, core, noisy, helpful in [('2023-24', 2., 1.8, 1.99), ('2024-25', 2., 2.01, 1.995)]:
            for package, score in [('absence_core', core), ('noisy', noisy), ('helpful', helpful)]:
                rows.append({'season': season, 'package': package, 'mae': score, 'feature_count': 5})
        self.assertEqual(select_package(rows), 'helpful')
        self.assertEqual(select_package([r for r in rows if r['package'] != 'helpful']), 'absence_core')

    def test_packages_omit_clusters_and_tracking(self):
        for stat in ['points', 'rebounds', 'assists']:
            for columns in packages(stat).values():
                self.assertEqual(len(columns), len(set(columns)))
                self.assertFalse(any('cluster' in c or 'lag_track' in c for c in columns))

    def test_calibration_and_assessment_exclude_proxy_pushes_and_report_intervals(self):
        actual = np.tile([6., 4.], 20)
        cal = pd.DataFrame({'actual': actual, 'prediction': actual, 'line': 5., 'scale': 1.})
        classifier, quantiles, rate = calibration_fit(cal)
        test = pd.DataFrame({'playerid': [1, 2, 3], 'season': '2025-26',
                             'date': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03']),
                             'team': 'A', 'opponent': 'B', 'points': [6., 4., 5.],
                             'points_avg_season': 5., 'points_scale': 1.})
        result, output = assess(test, np.array([6., 4., 5.]), 'points', classifier, quantiles, rate)
        self.assertEqual(result['mae'], 0)
        self.assertEqual(result['direction_confidence']['nonpush_rows'], 2)
        self.assertEqual(result['direction_confidence']['accuracy'], 1)
        self.assertEqual(result['intervals'][-1]['observed_coverage'], 1)
        self.assertEqual(len(output), 3)
        self.assertEqual(sum(b['count'] for b in result['direction_confidence']['buckets']), 2)


if __name__ == '__main__':
    unittest.main()
