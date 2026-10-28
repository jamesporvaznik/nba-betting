import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from models.data.load import TRAINING_SEASONS
from models.features.selected import selected_feature_cols
from models.train import run
from models.training.blend import SelectedBlendRegressor
from models.training.selected import train_selected


class SelectedBlendTests(unittest.TestCase):
    def test_serialization_column_order_and_missing_features(self):
        x = pd.DataFrame({'a': np.arange(20.), 'b': np.arange(20.) % 3})
        model = SelectedBlendRegressor(iterations=3).fit(x, 2*x.a+x.b)
        expected = model.predict(x)
        np.testing.assert_allclose(expected, model.predict(x[['b','a']]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_allclose(expected,joblib.load(path).predict(x))
        with self.assertRaises(ValueError):
            model.predict(x[['a']])

    def test_fixed_feature_contract(self):
        for stat,count in [('points',42),('rebounds',68),('assists',48)]:
            columns=selected_feature_cols(stat)
            self.assertEqual(len(columns),count)
            self.assertEqual(len(set(columns)),count)
            self.assertFalse(any('cluster' in c or 'retrospective' in c for c in columns))


class SelectedTrainingTests(unittest.TestCase):
    def test_training_only_filters_test_before_engineering_or_fitting(self):
        rows=[]
        for season,year in [('2022-23',2023),('2023-24',2024),('2024-25',2025),('2025-26',2026)]:
            for day in range(1,7):
                row={c:1. for c in selected_feature_cols('points')}
                row.update(playerid=1,season=season,date=pd.Timestamp(year,1,day),team='A',opponent='B',
                           points=1. if day%2 else 3.,points_avg_season=2.,points_scale=1.,
                           points_avg5=1.5 if day%2 else 2.5,prior_games_season=20,minutes_avg_season=30.)
                if season=='2025-26':
                    row['points_avg5']=99999.
                rows.append(row)
        frame=pd.DataFrame(rows)
        snapshot={'players':[{'season':s} for s in TRAINING_SEASONS+['2025-26']],
                  'teams':[], 'absences':[], 'positions':[]}
        seen=[]
        class RecordingModel:
            def fit(self,x,y):
                seen.append(x.copy())
                return self
            def predict(self,x):
                return x.iloc[:,0].to_numpy()
            def get_params(self):
                return {}
        def build(snap):
            seasons={r['season'] for r in snap['players']}
            self.assertEqual(seasons,set(TRAINING_SEASONS))
            return frame[frame.season.isin(seasons)].copy()
        with patch('models.training.selected.build_selected_features',side_effect=build), \
             patch('models.training.selected.SelectedBlendRegressor',RecordingModel), \
             contextlib.redirect_stdout(io.StringIO()):
            results,report,predictions,calibration,used=train_selected(snapshot,stats=['points'])
        self.assertFalse(report['test_used'])
        self.assertIsNone(report['stats']['points']['evaluation'])
        self.assertEqual(predictions,{})
        self.assertEqual(len(seen),3)
        self.assertTrue(all(not x.iloc[:,0].eq(99999).any() for x in seen))
        self.assertTrue((pd.to_datetime(calibration['points'].training_last_date)<calibration['points'].date).all())

    def test_default_load_excludes_test_and_no_save_writes_nothing(self):
        with patch('models.train.capture_training_snapshot',return_value={}) as capture, \
             patch('models.train.train_selected',return_value=({}, {}, {}, {}, {})), \
             patch('pathlib.Path.mkdir',side_effect=AssertionError('No directory writes allowed')), \
             patch('pathlib.Path.write_text',side_effect=AssertionError('No file writes allowed')), \
             contextlib.redirect_stdout(io.StringIO()):
            run(save=False)
        capture.assert_called_once_with(TRAINING_SEASONS)

    def test_explicit_baseline_dispatch_and_invalid_options(self):
        with self.assertRaises(ValueError):
            run(model='baseline',save=False)
        with self.assertRaises(ValueError):
            run(save=False,output_dir='/tmp/unused-model-directory')
        with patch('models.training.legacy.run_baselines',return_value={'points':{}}) as old:
            result=run(stats=['points'],model='baseline',evaluate=True,save=False,snapshot={'players':[]})
        self.assertEqual(result,{'points':{}})
        old.assert_called_once()


if __name__=='__main__':
    unittest.main()
