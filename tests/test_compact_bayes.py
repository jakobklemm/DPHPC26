"""Check numerical equivalence and wide-feature storage without large allocations."""
import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge

from ml.models.compact_bayes import CompactBayesianRidge
from ml.models.ridge_regression import PerformanceModel
from ml.models.ridge_regression_three_way import ThreeWayPerformanceModel


class CompactBayesTests(unittest.TestCase):
    def test_matches_dense_estimator(self):
        for shape in [(8, 30), (30, 8), (8, 8)]:
            for intercept in (True, False):
                for weighted in (True, False):
                    with self.subTest(shape=shape, intercept=intercept, weighted=weighted):
                        rng = np.random.default_rng(4)
                        X = rng.normal(size=shape) + 2
                        X[:, -1] = X[:, 0]  # rank deficient, including unseen directions
                        y = rng.normal(size=shape[0])
                        query = rng.normal(size=(5, shape[1]))
                        weights = rng.uniform(.1, 2, shape[0]) if weighted else None
                        options = dict(fit_intercept=intercept, compute_score=True)
                        dense = BayesianRidge(**options).fit(X, y, sample_weight=weights)
                        compact = CompactBayesianRidge(**options).fit(X, y, sample_weight=weights)
                        np.testing.assert_allclose(compact.predict(query, return_std=True),
                                                   dense.predict(query, return_std=True), rtol=1e-6, atol=1e-8)
                        np.testing.assert_allclose(compact.scores_, dense.scores_, rtol=1e-6, atol=1e-8)
                        np.testing.assert_allclose(compact.posterior_variance(query),
                                                   np.einsum('ij,jk,ik->i', query, dense.sigma_, query),
                                                   rtol=1e-6, atol=1e-8)

    def test_wider_than_lapack_square_limit(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(6, 50000))
        model = CompactBayesianRidge().fit(X, rng.normal(size=6))
        self.assertEqual(model._basis.shape, (6, 50000))
        self.assertEqual(model._variance.shape, (6,))
        self.assertFalse(hasattr(model, 'sigma_'))
        self.assertTrue(np.isfinite(model.predict(X[:2], return_std=True)).all())

    def test_serialization_clone_and_refit(self):
        X = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 0., 1.]})
        model = CompactBayesianRidge(tol=1e-5).fit(X, [1., 3., 2.])
        self.assertEqual(clone(model).tol, 1e-5)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_allclose(joblib.load(stream).predict(X, return_std=True),
                                   model.predict(X, return_std=True))
        model.fit(np.ones((4, 3)), [1., 2., 3., 4.])
        self.assertEqual(model.n_features_in_, 3)
        self.assertTrue(np.isfinite(model.predict(np.ones((2, 3)), return_std=True)).all())

    def test_ridge_models_and_cost_reporting(self):
        table = pd.DataFrame({'application': ['llama.cpp'] * 8,
                              'cfg.x': ['a', 'b'] * 4,
                              'cfg.y': ['a', 'a', 'b', 'b'] * 2,
                              'hw.cores': [2] * 4 + [4] * 4,
                              'relative': [.7, .8, .9, 1., .6, .9, .8, 1.]})
        for kind in (PerformanceModel, ThreeWayPerformanceModel):
            model = kind().fit(table)
            self.assertIsInstance(model.regressor, CompactBayesianRidge)
            costs = model._costs({'cores': 2}, {'x': 'b', 'y': 'b'}, [{'x': 'a', 'y': 'a'}])
            self.assertTrue(np.isfinite(costs).all())


if __name__ == '__main__':
    unittest.main()
