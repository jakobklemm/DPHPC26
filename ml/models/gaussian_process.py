"""Gaussian process performance surrogate.

Technical description
- Bayesian kernel regression on log-relative performance with a constant times
  Matérn covariance plus a learned white-noise term; native predictive
  standard deviation supports exploration.
- Length scale bounded to [0.1, 100]: one changed flag is ~1.4 apart in the
  scaled one-hot space, so the floor rules out the degenerate "every build
  uncorrelated" fit that small samples otherwise sometimes reach.
- ConvergenceWarning "noise_level is close to the specified lower bound" is
  expected with few distinct builds (exact interpolation is the likelihood
  optimum) and is left visible on purpose.

Determinism
- Predictions are deterministic after fitting. Kernel optimizer restarts use
  randomness; fixed random_state makes training reproducible.

Adjustable parameters
- kernel (incl. WhiteKernel noise bounds), alpha (numerical jitter), normalize_y, optimizer,
  n_restarts_optimizer and random_state. No bootstrap ensemble is fitted.

Benefits
- Nonlinear small-data modeling with native model-dependent uncertainty.

Drawbacks
- One-hot distances may poorly describe mixed hardware and configuration input;
  exact fitting scales cubically and high dimensionality weakens similarity.
- Native uncertainty is not a guarantee of calibration or hardware transfer.

Overfitting risk
- Very small noise and short kernel length scales can fit measurement noise.
  Use restrained kernels and assess predictions on whole held-out machines.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .common import RegressionPerformanceModel


class GaussianProcessPerformanceModel(RegressionPerformanceModel):
    # Scaled numeric distances and native GP uncertainty
    scale_features = True
    native_uncertainty = True
    uncertainty_method = "GP predictive standard deviation (model-dependent)"

    def _make_regressor(self, seed, n_features):
        # Matérn similarity, learned noise; length-scale floor against collapse
        # Target normalization and seeded optimizer restarts
        kernel = (ConstantKernel(1.0)
                  * Matern(length_scale=1.0, length_scale_bounds=(0.1, 100.0), nu=1.5)
                  + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0)))
        params = dict(kernel=kernel, normalize_y=True, alpha=1e-8, n_restarts_optimizer=2,
                      random_state=seed)
        # Constructor overrides for kernel, noise and optimizer
        params.update(self.estimator_params)
        return GaussianProcessRegressor(**params)

    def _costs(self, hardware, best, variants):
        """Change to the best build, uncertainty from the joint predictive covariance."""
        frame = self._prediction_frame(hardware, [best] + variants)
        log_prediction = self.regressor.predict(frame)
        if self.regressor.constant:
            log_std = np.zeros(len(variants))
        else:
            X = self.regressor.encoder.transform(frame)
            _, cov = self.regressor.regressor.predict(X, return_cov=True)
            variance = np.diag(cov)[1:] + cov[0, 0] - 2 * cov[0, 1:]
            log_std = np.sqrt(np.clip(variance, 0, None))
        return np.exp(log_prediction[1:] - log_prediction[0]) - 1, np.exp(log_std) - 1
