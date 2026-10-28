"""Small multilayer perceptron performance surrogate.

Technical description
- Fully connected neural regressor on log-relative performance with two small
  hidden layers; learns nonlinear feature combinations from tabular inputs.

Determinism
- Weight initialization and bootstrap resampling are stochastic. Fixed seeds
  make fits reproducible within the same numerical/software environment.

Adjustable parameters
- hidden_layer_sizes, activation, solver, alpha, max_iter, tol, learning_rate_init
  (for stochastic solvers), random_state and shared ensemble_size.

Benefits
- Simple neural baseline without spatial assumptions or manual interactions.

Drawbacks
- Scaling, architecture and convergence require care; bootstrap uncertainty is
  uncalibrated. A neural network is not automatically better for small tables.
  
Overfitting risk
- Even small networks can memorize a few hundred runs. Small hidden layers,
  L2 regularization and held-out-machine evaluation help constrain that risk.
"""

from sklearn.neural_network import MLPRegressor

from .common import RegressionPerformanceModel


class SmallMLPPerformanceModel(RegressionPerformanceModel):
    # Scaled numeric inputs alongside one-hot build settings
    scale_features = True

    def _make_regressor(self, seed, n_features):
        # Two small hidden layers with L2 regularization
        # Full-sample L-BFGS; seeded initial weights
        params = dict(hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
                      alpha=0.01, max_iter=2000, random_state=seed)
        # Architecture/solver overrides and shared bootstrap uncertainty
        params.update(self.estimator_params)
        return MLPRegressor(**params)
