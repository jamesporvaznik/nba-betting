import unittest

import numpy as np
import pandas as pd

from models.retest_blend_features import development_only, inner_calibration_split, choose_addition, fit_blend


class BlendFeatureRetestTests(unittest.TestCase):
    def test_test_season_is_removed_from_all_outcome_tables(self):
        records = [{'season':'2022-23'}, {'season':'2023-24'}, {'season':'2024-25'}, {'season':'2025-26'}]
        source = {key:list(records) for key in ['players','teams','absences']}
        source['positions'] = [{'nba_api_id':1,'position':'Guard'}]
        result = development_only(source)
        for key in ['players','teams','absences']:
            self.assertEqual({r['season'] for r in result[key]}, {'2022-23','2023-24','2024-25'})
            self.assertEqual(len(source[key]),4)

    def test_calibration_keeps_whole_dates_strictly_after_inner_training(self):
        dates = pd.date_range('2024-01-01',periods=20)
        frame = pd.DataFrame({'date':np.repeat(dates,2)})
        train, calibration = inner_calibration_split(frame)
        self.assertLess(train.date.max(),calibration.date.min())
        self.assertEqual(len(train),30)
        self.assertEqual(len(calibration),10)

    def test_selection_requires_both_folds_and_minimum_improvement(self):
        rows = [{'stat':'points','variant':name,'fold':fold,'mae':mae}
                for name, values in [('baseline',[5.,5.]),('mixed',[4.9,5.01]),('tiny',[4.999,4.999]),('pace',[4.99,4.98])]
                for fold,mae in zip(['a','b'],values)]
        self.assertEqual(choose_addition(rows,'points'),'pace')
        self.assertEqual(choose_addition([r for r in rows if r['variant']!='pace'],'points'),'baseline')

    def test_imputation_and_forecasts_do_not_learn_from_validation_batch_or_labels(self):
        train = pd.DataFrame({'x':np.arange(20,dtype=float),'points':np.arange(20,dtype=float)})
        validation = pd.DataFrame({'x':[np.nan,5.],'points':[0.,0.]})
        config = {'tree_weight':.75,'params':{'loss_function':'MAE','iterations':3,'depth':2,'learning_rate':.04,'l2_leaf_reg':10}}
        original = fit_blend(train,validation,['x'],'points',config)
        augmented = pd.concat([validation.assign(points=999),pd.DataFrame({'x':[10000.,20000.,30000.],'points':[0.,0.,0.]})],ignore_index=True)
        after = fit_blend(train,augmented,['x'],'points',config)
        np.testing.assert_allclose(original,after[:2])


if __name__=='__main__':
    unittest.main()
