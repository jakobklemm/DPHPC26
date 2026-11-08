"""Experiment-only Bayesian ridge acceleration used in the exhaustive replay.

Keep sklearn's fitting algorithm, but represent the posterior covariance using
its reduced singular vectors. This avoids allocating a feature-by-feature matrix.
The adapter checks sklearn's source layout because it relies on private details.
"""
import ast
import inspect
import textwrap

import numpy as np
from sklearn.linear_model import BayesianRidge, _bayes


class ReducedFit(ast.NodeTransformer):
    def visit_Assign(self, node):
        covariance = any(isinstance(target, ast.Attribute) and target.attr == 'sigma_'
                         for target in node.targets)
        if covariance:
            return ast.parse(
                'self._basis = Vh.copy()\n'
                'self._squared_singular_values = eigen_vals_.copy()'
            ).body
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == 'full_matrices':
            node.value = ast.Constant(False)
        return self.generic_visit(node)


def reduced_fit():
    source = textwrap.dedent(inspect.getsource(BayesianRidge.fit))
    if 'Vh_full' not in source or 'self.sigma_ = np.dot(' not in source:
        raise RuntimeError('Unsupported sklearn covariance implementation. '
                           'Run without --compact-bayes to use standard sklearn.')
    method = ast.fix_missing_locations(ReducedFit().visit(ast.parse(source)))
    namespace = dict(vars(_bayes))
    exec(compile(method, '<compact Bayesian fit>', 'exec'), namespace)
    return namespace['fit']


class CompactBayesianRidge(BayesianRidge):
    fit = reduced_fit()

    def predict(self, X, return_std=False):
        mean = super().predict(X)
        if not return_std:
            return mean
        matrix = np.asarray(X)
        projected = matrix @ self._basis.T
        eigenvalues = self.alpha_ * self._squared_singular_values
        weights = eigenvalues / (self.lambda_ + eigenvalues)
        variance = (np.square(matrix).sum(axis=1)
                    - (np.square(projected) * weights).sum(axis=1)) / self.lambda_
        return mean, np.sqrt(np.maximum(variance, 0) + 1 / self.alpha_)


def activate():
    """Apply the faster estimator only inside this experiment's Python process."""
    from ml.models import ridge_regression, bayes_pairs
    ridge_regression.BayesianRidge = CompactBayesianRidge
    bayes_pairs.BayesianRidge = CompactBayesianRidge
