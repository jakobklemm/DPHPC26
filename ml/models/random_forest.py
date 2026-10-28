"""Random forest performance surrogate.

Technical description
- Average randomized regression trees fitted to log-relative performance.
- Learn conditional hardware/flag relationships without explicit interactions.

Determinism
- Tree construction and bootstrap uncertainty are stochastic; fixed seeds make
  CPU fits reproducible within the same software environment.

Adjustable parameters
- n_estimators, max_depth, min_samples_leaf, max_features, bootstrap, n_jobs;
  random_state and ensemble_size control shared reproducibility/uncertainty.

Benefits
- Capture nonlinear thresholds and flag interactions with little preprocessing.
Drawbacks
- Weak extrapolation to unfamiliar hardware; ensembles increase memory and cost.
- Bootstrap spread is uncalibrated and can miss shared model bias.

Overfitting risk
- Deep trees can memorize sparse hardware/configuration combinations. Restrict
  depth and leaf size and evaluate on held-out machines with limited data.
"""

from sklearn.ensemble import RandomForestRegressor

from .common import RegressionPerformanceModel


class RandomForestPerformanceModel(RegressionPerformanceModel):
    def _make_regressor(self, seed, n_features):
        # Depth and leaf-size limits for small datasets
        # Seeded tree construction
        params = dict(n_estimators=200, max_depth=8, min_samples_leaf=2,
                      random_state=seed, n_jobs=1)
        # Constructor overrides for estimator defaults
        # Shared fitting, bootstrap uncertainty and build selection
        params.update(self.estimator_params)
        return RandomForestRegressor(**params)
