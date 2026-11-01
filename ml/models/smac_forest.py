"""SMAC-style random forest surrogate for small active-sampling budgets.

Technical description
- Random forest on log-relative performance, as in SMAC (Hutter et al., 2011).
- Uncertainty is the spread between the individual trees, not bootstrap refits
  of a whole forest, so one fit is enough.

Determinism
- Tree construction is stochastic; a fixed random_state makes CPU fits
  reproducible within the same software environment.

Adjustable parameters
- n_estimators, min_samples_split, min_samples_leaf, max_features, n_jobs and
  random_state. No bootstrap ensemble is fitted.

Benefits
- Handles categorical flags and interactions without kernels or feature design.
- About 10 times faster per fit than random-forest, which refits 9 forests.
Drawbacks
- Tree spread is uncalibrated and shrinks far from the data only weakly.
- Weak extrapolation to option combinations unlike the measured builds.

Overfitting risk
- Deep trees follow noise with about 25 builds. Prediction averaging limits
  this; trust measured builds over predictions for the final choice.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .common import RegressionPerformanceModel


class TreeSpreadForest(RandomForestRegressor):
    """Random forest that reports the spread of its trees as uncertainty."""

    def predict(self, X, return_std=False):
        draws = np.array([tree.predict(np.asarray(X)) for tree in self.estimators_])
        return (draws.mean(axis=0), draws.std(axis=0)) if return_std else draws.mean(axis=0)


class SmacForestPerformanceModel(RegressionPerformanceModel):
    native_uncertainty = True
    uncertainty_method = "spread between trees (uncalibrated)"

    def _make_regressor(self, seed, n_features):
        # SMAC-like: splits on most features, leaves of single builds allowed
        params = dict(n_estimators=100, min_samples_split=3, min_samples_leaf=1,
                      max_features=5 / 6, random_state=seed, n_jobs=1)
        params.update(self.estimator_params)
        return TreeSpreadForest(**params)
