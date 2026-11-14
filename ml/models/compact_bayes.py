"""Bayesian ridge with reduced SVD and implicit posterior covariance.

Retain sklearn's evidence updates and stopping rule in original coordinates.
Directions outside the training row space retain prior variance 1 / lambda.
"""
import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import BayesianRidge
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data


class CompactBayesianRidge(BayesianRidge):
    """BayesianRidge parameters and predictions, with compact covariance storage.

    Use ``posterior_variance`` for quadratic covariance forms; ``sigma_`` is
    intentionally not materialized. Coefficients remain in original coordinates.
    """

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], y_numeric=True)
        n_samples, n_features = X.shape
        weights = (None if sample_weight is None else
                   _check_sample_weight(sample_weight, X, dtype=X.dtype))
        sw_sum = n_samples if weights is None else weights.sum()
        y_mean = np.average(y, weights=weights)
        y_var = np.average((y - y_mean) ** 2, weights=weights)
        offset = np.average(X, axis=0, weights=weights) if self.fit_intercept else np.zeros(n_features, dtype=X.dtype)
        y_offset = y_mean if self.fit_intercept else 0.
        X = X - offset
        y = y - y_offset
        if weights is not None:
            root_weights = np.sqrt(weights)
            X *= root_weights[:, None]
            y *= root_weights
        self.X_offset_ = offset
        self.X_scale_ = np.ones(n_features, dtype=X.dtype)
        alpha = np.asarray(1 / (y_var + np.finfo(np.float64).eps)
                           if self.alpha_init is None else self.alpha_init, dtype=X.dtype)
        precision = np.asarray(1. if self.lambda_init is None else self.lambda_init, dtype=X.dtype)
        U, singular, basis = svd(X, full_matrices=False)
        eigenvalues = singular ** 2
        XT_y = X.T @ y
        previous = None
        self.scores_ = []
        for iteration in range(self.max_iter):
            coef, sse = self._update_coef_(X, y, n_samples, n_features, XT_y,
                                           U, basis, eigenvalues, alpha, precision)
            if self.compute_score:
                self.scores_.append(self._log_marginal_likelihood(
                    n_samples, n_features, sw_sum, eigenvalues, alpha, precision, coef, sse))
            gamma = np.sum(alpha * eigenvalues / (precision + alpha * eigenvalues))
            precision = (gamma + 2 * self.lambda_1) / (np.sum(coef ** 2) + 2 * self.lambda_2)
            alpha = (sw_sum - gamma + 2 * self.alpha_1) / (sse + 2 * self.alpha_2)
            if previous is not None and np.sum(np.abs(previous - coef)) < self.tol:
                if self.verbose:
                    print('Convergence after ', iteration, ' iterations')
                break
            previous = coef.copy()
        self.n_iter_ = iteration + 1
        self.alpha_, self.lambda_ = alpha, precision
        self.coef_, sse = self._update_coef_(X, y, n_samples, n_features, XT_y,
                                            U, basis, eigenvalues, alpha, precision)
        if self.compute_score:
            # Match sklearn's final diagnostic, which uses the last loop coef.
            self.scores_.append(self._log_marginal_likelihood(
                n_samples, n_features, sw_sum, eigenvalues, alpha, precision, coef, sse))
            self.scores_ = np.asarray(self.scores_)
        self._basis = basis
        self._variance = 1 / (precision + alpha * eigenvalues)
        self.intercept_ = y_offset - offset @ self.coef_
        return self

    def posterior_variance(self, X):
        """Return diag(X covariance X.T), excluding observation noise."""
        check_is_fitted(self, '_variance')
        X = np.asarray(X)
        projected = X @ self._basis.T
        variance = np.square(projected) @ self._variance
        # Explicit residual avoids cancellation for vectors near the row space.
        residual = X - projected @ self._basis
        variance += np.einsum('ij,ij->i', residual, residual) / self.lambda_
        return np.maximum(variance, 0)

    def predict(self, X, return_std=False):
        mean = super().predict(X)
        if not return_std:
            return mean
        return mean, np.sqrt(self.posterior_variance(X) + 1 / self.alpha_)
