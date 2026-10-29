"""Step 2: the surrogate model.

Why predict performance and not directly a build: a classifier (hardware ->
build) needs a correct answer per example, so it could only learn from the
fastest builds and would throw the slow ones away. Predicting performance,
every run is a valid example (the slow ones show what to avoid), and the
best build is then found by comparing the predictions.

A surrogate model replaces an expensive measurement with a cheap prediction.
Here it answers: "how fast would this build be on this machine?" without
building and benchmarking it.
    (hardware features, build configuration)  ->  predicted relative performance

Choosing a build for a machine: predict the performance of every candidate
build on that machine and take the one with the highest prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from ..load import DEFAULT, HW, columns, strip
from .common import BaseModel


def make_regressor():
    """Regularize weakly supported effects and provide predictive uncertainty."""
    return BayesianRidge()


class PerformanceModel(BaseModel):
    """Bayesian ridge surrogate with build-option by hardware interactions."""

    def fit(self, table):
        self.hw_features = [strip(c) for c in columns(table, HW)]
        self._fit_options(table)

        # Input: hardware features + build options of every build. Output: the log of its relative performance,
        # so that effects multiply: -0.12 in log space is about 12 % slower, whatever the base performance.
        X = self._encode(table.rename(columns=strip)[self.hw_features + self.options], fit=True)
        y = np.log(table["relative"])
        self.regressor = make_regressor().fit(X, y)

        # How well the model fits
        self.r2 = self.regressor.score(X, y)
        self.noise = np.exp(np.sqrt(1 / self.regressor.alpha_)) - 1
        return self

    def _encode(self, data, fit=False):
        """
        Turn the inputs into numbers

        True/False become 1/0. Text (e.g. gpu_vendor, or an option like GGML_LTO=ON) becomes one 0/1 column per value, e.g. "GGML_LTO_ON"
        
        Interactions: every option column times every hardware column so that the effect of an option can differ between machines
        """
        X = pd.get_dummies(data.astype({c: float for c in data.select_dtypes(bool).columns}), dtype=float)
        if fit:
            reference = {f"{o}_{self.values[o][0]}" for o in self.options}
            option_values = {f"{o}_{v}" for o in self.options for v in self.values[o]}
            kept = [c for c in X.columns if X[c].nunique() > 1 and c not in reference]
            self.option_columns = [c for c in kept if c in option_values]
            self.hw_columns = [c for c in kept if c not in option_values]
        X = X.reindex(columns=self.option_columns + self.hw_columns, fill_value=0)
        # All interaction columns built first and added at once, one insert per column fragments the frame
        interactions = {f"{o} x {h}": X[o] * X[h] for o in self.option_columns for h in self.hw_columns}
        X = pd.concat([X, pd.DataFrame(interactions, index=X.index)], axis=1)
        if X.shape[1] == 0:
            # A baseline may have no varying options and no hardware features.
            X["constant"] = 1.0
        return X

    def predict_performance(self, hardware, configs):
        rows = pd.DataFrame([{**hardware, **{o: c.get(o, DEFAULT) for o in self.options}} for c in configs])
        X = self._encode(rows[self.hw_features + self.options])
        log_mean, log_std = self.regressor.predict(X, return_std=True)
        mean = np.exp(log_mean)
        return mean, mean * log_std

    def _costs(self, hardware, best, variants):
        """Exact for ridge: the change is linear in the encoded inputs, its
        uncertainty follows from the coefficient covariance."""
        builds = pd.DataFrame([{**hardware, **c} for c in [best] + variants])
        X = self._encode(builds[self.hw_features + self.options]).to_numpy()
        diff = X[1:] - X[0]
        log_change = diff @ self.regressor.coef_
        log_std = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.regressor.sigma_, diff))
        return np.exp(log_change) - 1, np.exp(log_std) - 1
