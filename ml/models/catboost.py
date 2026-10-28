"""CatBoost performance surrogate.

Technical description
- Sequential gradient-boosted trees on log-relative performance, using native
  categorical hardware descriptors and build settings

Determinism
- Boosting randomization and bootstrap uncertainty are stochastic
- seeded, single-threaded CPU fits are reproducible in the same software environment
- GPU training is not guaranteed deterministic

Adjustable parameters
- iterations, depth, learning_rate, l2_leaf_reg, loss_function, thread_count,
  task_type, random_seed; shared random_state and ensemble_size are supported.

Benefits
- Native categorical handling and automatic nonlinear flag/hardware effects.
Drawbacks
- Extra dependency and tuning; no guarantee of transfer to unseen hardware.
- Shared bootstrap spread is uncalibrated, not CatBoost's native noise estimate.

Overfitting risk
- Repeated corrections can fit noise in a few hundred runs. Shallow trees and
  L2 regularization reduce risk; verify choices on whole held-out machines.
"""

from .common import RegressionPerformanceModel


class CatBoostPerformanceModel(RegressionPerformanceModel):
    # Native category strings without one-hot encoding
    native_categories = True

    def _make_regressor(self, seed, n_features):
        # Lazy dependency import on CatBoost selection
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost requires catboost; install with "
                              "'python -m pip install -r ml/requirements.txt'") from exc
        # Shallow regularized trees; seeded single-threaded CPU fits
        params = dict(iterations=300, depth=4, learning_rate=0.03,
                      l2_leaf_reg=5, loss_function="RMSE", task_type="CPU",
                      random_seed=seed, thread_count=1, verbose=False,
                      allow_const_label=True)
        # Constructor overrides and shared bootstrap uncertainty
        params.update(self.estimator_params)
        # Auxiliary file writing disabled despite constructor overrides
        params["allow_writing_files"] = False
        return CatBoostRegressor(**params)
