"""Support vector regression performance surrogate.

Technical description
- Epsilon-insensitive regression on log-relative performance, using an RBF
  kernel and a regularized function supported by selected training examples.

Determinism
- The underlying estimator is deterministic for fixed inputs and parameters.
- Shared bootstrap resampling is stochastic but reproducible with a fixed seed.

Adjustable parameters
- C, epsilon, kernel, gamma, degree, coef0, tol and max_iter; random_state
  and ensemble_size control shared bootstrap uncertainty.

Benefits
- Regularized nonlinear regression suitable for a few hundred observations;
  epsilon ignores small target deviations inside its tolerance band.

Drawbacks
- Scaling and hyperparameters matter; no native predictive uncertainty and
  kernel similarity may not transfer across hardware. Bootstrap adds cost.
  
Overfitting risk
- Large C, large gamma and small epsilon can fit noise. Validate regularization
  and kernel width using held-out machines and limited tuning.
"""

from sklearn.svm import SVR

from .common import RegressionPerformanceModel


class SupportVectorPerformanceModel(RegressionPerformanceModel):
    # Scaled numeric inputs for balanced RBF distances
    scale_features = True

    def _make_regressor(self, seed, n_features):
        # C error penalty; epsilon tolerance in log-performance space
        params = dict(kernel="rbf", C=1.0, epsilon=0.02, gamma="scale")
        # Constructor tuning and shared bootstrap uncertainty
        params.update(self.estimator_params)
        return SVR(**params)
