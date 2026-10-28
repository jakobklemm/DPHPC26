"""Kernel ridge performance surrogate.

Technical description
- L2-regularized regression in a kernel feature space; an RBF kernel learns
  nonlinear relationships in log-relative performance.

Determinism
- The underlying estimator is deterministic for fixed inputs and parameters.
- Shared bootstrap resampling is stochastic but reproducible with a fixed seed.

Adjustable parameters
- alpha, kernel, gamma, degree, coef0 and kernel_params; random_state and
  ensemble_size control shared bootstrap uncertainty.

Benefits
- Compact nonlinear extension of ridge, suited to modest training tables.
Drawbacks
- Requires feature scaling and meaningful distances; dense kernel storage and
  solving grow with the dataset. Uncertainty needs uncalibrated bootstrap fits.
  
Overfitting risk
- Small alpha and large gamma can memorize runs; strong regularization may
  underfit. Tune on training machines, not the held-out hardware.
"""

from sklearn.kernel_ridge import KernelRidge

from .common import RegressionPerformanceModel


class KernelRidgePerformanceModel(RegressionPerformanceModel):
    # Comparable numeric hardware scales for kernel distances
    scale_features = True

    def _make_regressor(self, seed, n_features):
        # RBF regularization and feature-count-based kernel width
        params = dict(kernel="rbf", alpha=1.0, gamma=1.0 / n_features)
        # Constructor tuning and shared bootstrap uncertainty
        params.update(self.estimator_params)
        return KernelRidge(**params)
