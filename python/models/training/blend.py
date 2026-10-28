"""Reusable selected blend, fitted only on explicitly supplied training rows."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class SelectedBlendRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, tree_weight=.75, iterations=600, depth=4, learning_rate=.04,
                 l2_leaf_reg=10, random_seed=42):
        self.tree_weight = tree_weight
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def _matrix(self, frame, fitting=False):
        if isinstance(frame, pd.DataFrame):
            if frame.columns.duplicated().any():
                raise ValueError('Duplicate input feature columns')
            if fitting:
                self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            elif hasattr(self, 'feature_names_in_'):
                missing = set(self.feature_names_in_) - set(frame.columns)
                if missing:
                    raise ValueError(f'Missing prediction features: {sorted(missing)}')
                frame = frame.loc[:, self.feature_names_in_]
        matrix = np.asarray(frame, dtype=float)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError('Features must be a nonempty finite matrix')
        if not fitting and matrix.shape[1] != self.n_features_in_:
            raise ValueError('Prediction feature count differs from training')
        return matrix

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        if not 0 <= self.tree_weight <= 1:
            raise ValueError('Tree weight must be between zero and one')
        # Clear names when refitting an existing estimator on an unnamed matrix.
        if hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        matrix = self._matrix(X, fitting=True)
        target = np.asarray(y, dtype=float)
        if target.shape != (len(matrix),) or not np.isfinite(target).all():
            raise ValueError('Targets must match training rows and be finite')
        self.n_features_in_ = matrix.shape[1]
        self.linear_model_ = LinearRegression().fit(matrix, target)
        self.tree_model_ = CatBoostRegressor(
            loss_function='MAE', iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed, thread_count=1, verbose=False,
            allow_writing_files=False).fit(matrix, target)
        return self

    def predict(self, X):
        check_is_fitted(self, ['linear_model_', 'tree_model_', 'n_features_in_'])
        matrix = self._matrix(X)
        return ((1-self.tree_weight) * self.linear_model_.predict(matrix) +
                self.tree_weight * self.tree_model_.predict(matrix))
