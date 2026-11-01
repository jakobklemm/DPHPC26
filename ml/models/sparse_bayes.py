"""Sparse Bayesian linear surrogate (ARD) for active sampling.

Technical description
- Automatic relevance determination on log-relative performance: each encoded
  input gets its own prior, and inputs without support are pruned to zero.
- Assumes that only a few options matter, as for llama.cpp on GPUs.

Determinism
- The estimator is deterministic for fixed inputs and parameters.

Adjustable parameters
- ARDRegression parameters (max_iter, tol, threshold_lambda, alpha_1, alpha_2,
  lambda_1, lambda_2). No bootstrap ensemble is fitted.

Benefits
- Ranks builds well from few measurements when the assumption holds, and the
  surviving options are a direct estimate of parameter importance.
Drawbacks
- No interactions. When many options have small effects (CPU builds), pruning
  depends strongly on the measured builds, so predictions vary between runs.

Overfitting risk
- A few builds can make an irrelevant option look decisive, with too little
  uncertainty. Prefer the best measured build over the predicted best.
"""

from sklearn.linear_model import ARDRegression

from .common import RegressionPerformanceModel, StandardizedTarget


class SparseBayesPerformanceModel(RegressionPerformanceModel):
    native_uncertainty = True
    uncertainty_method = "ARD predictive standard deviation (model-dependent)"

    def _make_regressor(self, seed, n_features):
        return StandardizedTarget(ARDRegression(**self.estimator_params))
