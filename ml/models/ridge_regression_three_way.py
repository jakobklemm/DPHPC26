"""Sample-space Bayesian ridge with build-option × build-option × hardware terms.

The additional interactions can overfit when benchmark coverage is limited.
"""

from itertools import islice

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array

from ..load import DEFAULT, HW, columns, strip
from .ridge_regression import make_regressor
from .ridge_regression_interactions import InteractionPerformanceModel


class _SampleSpaceBayesianRidge:
    """Fit Bayesian ridge from feature dot products between benchmark rows."""

    def fit(self, kernel, y):
        # Center the feature dot products to fit an unregularized intercept.
        self.kernel_mean = kernel.mean(axis=0)
        kernel -= self.kernel_mean
        kernel -= self.kernel_mean[:, None]
        kernel += self.kernel_mean.mean()
        eigenvalues, eigenvectors = eigh(kernel, overwrite_a=True)
        np.maximum(eigenvalues, 0, out=eigenvalues)

        target_mean = y.mean()
        centered = y - target_mean
        target = eigenvectors.T @ centered
        alpha = 1 / (np.mean(centered ** 2) + np.finfo(float).eps)
        precision = 1.
        prior = make_regressor()
        for _ in range(prior.max_iter):
            denominator = eigenvalues + precision / alpha
            shrinkage = eigenvalues / denominator
            effective_features = shrinkage.sum()
            coefficients_sq = np.sum(eigenvalues * (target / denominator) ** 2)
            residuals_sq = np.sum(((1 - shrinkage) * target) ** 2)
            next_precision = ((effective_features + 2 * prior.lambda_1)
                              / (coefficients_sq + 2 * prior.lambda_2))
            next_alpha = ((len(y) - effective_features + 2 * prior.alpha_1)
                          / (residuals_sq + 2 * prior.alpha_2))
            # Monitor the fitted precisions without expanding the coefficients.
            change = max(abs(next_alpha / alpha - 1), abs(next_precision / precision - 1))
            alpha, precision = next_alpha, next_precision
            if change < 1e-10:
                break

        denominator = eigenvalues + precision / alpha
        dual = target / denominator
        self.weights = eigenvectors @ dual
        self.intercept_ = target_mean - (self.kernel_mean - self.kernel_mean.mean()) @ self.weights
        self.posterior_factor = eigenvectors / np.sqrt(denominator)
        self.alpha_, self.lambda_ = alpha, precision
        self.r2_ = r2_score(y, target_mean + eigenvectors @ (eigenvalues * dual))
        return self

    @staticmethod
    def center_cross_kernel(cross):
        """Center the training side of the query-by-training dot products."""
        return cross - cross.mean(axis=1, keepdims=True)

    def posterior_variance(self, cross, squared_norm):
        """Coefficient uncertainty for features or feature differences."""
        projected = cross @ self.posterior_factor
        correction = np.einsum("ij,ij->i", projected, projected)
        return np.maximum(squared_norm - correction, 0) / self.lambda_


class ThreeWayPerformanceModel(InteractionPerformanceModel):
    """Let each learned option-pair effect vary with encoded hardware features.

    Retain individual options, hardware, option × hardware, and option × option
    terms. Add varying A × B × h products, with A and B from different options
    and h from hardware. Pure three-option interactions are not included.
    """

    prediction_batch_size = 512

    def fit(self, table):
        self.hw_features = [strip(c) for c in columns(table, HW)]
        self._fit_options(table)
        y = check_array(np.log(table["relative"]), ensure_2d=False)
        data = table.rename(columns=strip)
        options, hardware = self._input_arrays(data, fit=True)

        indices = {name: i for i, name in enumerate(self.option_columns)}
        pair_indices = [(indices[a], indices[b]) for a, b in self._cross_option_pairs()]
        self._pair_left, self._pair_right = np.asarray(pair_indices, dtype=int).reshape(-1, 2).T
        pairs = self._pair_values(options)
        varying = np.ptp(pairs, axis=0) > 0
        self._pair_left, self._pair_right = self._pair_left[varying], self._pair_right[varying]
        pairs = pairs[:, varying]

        # Constant training triples vanish when the kernel is centered. Keep
        # their mask so omitted triples also contribute no prior variance at prediction.
        self._varying_triples = np.empty((pairs.shape[1], hardware.shape[1]), dtype=bool)
        for i, values in enumerate(hardware.T):
            self._varying_triples[:, i] = np.ptp(pairs * values[:, None], axis=0) > 0

        self._train_options, self._train_hardware, self._train_pairs = options, hardware, pairs
        self.regressor = _SampleSpaceBayesianRidge().fit(self._kernel(options, hardware, pairs), y)
        self.r2 = self.regressor.r2_
        self.noise = np.expm1(np.sqrt(1 / self.regressor.alpha_))
        return self

    def _input_arrays(self, data, fit=False):
        inputs = self._encode_inputs(data[self.hw_features + self.options], fit=fit)
        return (inputs[self.option_columns].to_numpy(dtype=float),
                inputs[self.hw_columns].to_numpy(dtype=float))

    def _pair_values(self, options):
        return options[:, self._pair_left] * options[:, self._pair_right]

    def _kernel(self, options, hardware, pairs):
        # The dot product of pair × hardware features factors into two small
        # dot products. Apply the same identity to option × hardware.
        hardware_similarity = hardware @ self._train_hardware.T
        configuration_similarity = options @ self._train_options.T
        configuration_similarity += pairs @ self._train_pairs.T
        configuration_similarity *= 1 + hardware_similarity
        return configuration_similarity + hardware_similarity

    def _configuration_norm(self, options, hardware_sq, pairs):
        """Squared norm of the terms that depend on build options."""
        option_norm = np.square(options).sum(axis=1)
        pairs_sq = np.square(pairs)
        triple_norm = np.einsum("ij,ij->i", pairs_sq @ self._varying_triples, hardware_sq)
        return option_norm * (1 + hardware_sq.sum(axis=1)) + pairs_sq.sum(axis=1) + triple_norm

    def predict_performance(self, hardware, configs):
        if not isinstance(self.regressor, _SampleSpaceBayesianRidge):
            return super().predict_performance(hardware, configs)

        means, spreads = [], []
        configs = iter(configs)
        while batch := list(islice(configs, self.prediction_batch_size)):
            rows = pd.DataFrame([{**hardware, **{o: c.get(o, DEFAULT) for o in self.options}} for c in batch])
            options, encoded_hardware = self._input_arrays(rows)
            pairs = self._pair_values(options)
            cross = self.regressor.center_cross_kernel(self._kernel(options, encoded_hardware, pairs))
            hardware_sq = np.square(encoded_hardware)
            squared_norm = self._configuration_norm(options, hardware_sq, pairs) + hardware_sq.sum(axis=1)
            if not (self.option_columns or self.hw_columns):
                # The original ridge encoding uses a constant feature in this case.
                squared_norm += 1
            variance = self.regressor.posterior_variance(cross, squared_norm)
            mean = np.exp(self.regressor.intercept_ + cross @ self.regressor.weights)
            means.append(mean)
            spreads.append(mean * np.sqrt(variance + 1 / self.regressor.alpha_))
        if not means:
            return np.empty(0), np.empty(0)
        return np.concatenate(means), np.concatenate(spreads)

    def _costs(self, hardware, best, variants):
        if not isinstance(self.regressor, _SampleSpaceBayesianRidge):
            return super()._costs(hardware, best, variants)

        rows = pd.DataFrame([{**hardware, **c} for c in [best] + variants])
        options, encoded_hardware = self._input_arrays(rows)
        # Hardware is fixed, so its main effects cancel in the difference.
        option_diff = options[1:] - options[0]
        pairs = self._pair_values(options)
        pair_diff = pairs[1:] - pairs[0]
        cross = self._kernel(options, encoded_hardware, pairs)
        cross = self.regressor.center_cross_kernel(cross[1:] - cross[0])
        squared_norm = self._configuration_norm(option_diff, np.square(encoded_hardware[1:]), pair_diff)
        return (np.expm1(cross @ self.regressor.weights),
                np.expm1(np.sqrt(self.regressor.posterior_variance(cross, squared_norm))))

    def _encode(self, data, fit=False):
        """Feature encoding for models saved before sample-space fitting."""
        X = super()._encode(data, fit=fit)
        terms = {f"{a} x {b} x {h}": X[a] * X[b] * X[h]
                 for a, b, h in self.hardware_option_pairs}
        return pd.concat([X, pd.DataFrame(terms, index=X.index)], axis=1)
