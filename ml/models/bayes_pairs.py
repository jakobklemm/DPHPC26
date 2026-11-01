"""Bayesian linear surrogate with all option pairs, for active sampling.

Technical description
- Bayesian ridge on log-relative performance over every encoded input and every
  pair of inputs. All terms share one Gaussian prior, which keeps a model with
  more coefficients than builds stable.

Determinism
- The estimator is deterministic for fixed inputs and parameters.

Adjustable parameters
- BayesianRidge parameters (max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2).
  No bootstrap ensemble is fitted.

Benefits
- Native uncertainty that grows for untried combinations, cheap refits, and
  pair effects that can be read off the coefficients.
Drawbacks
- Linear per term: no effects beyond pairs. One shared prior spreads weight over
  many small pair terms instead of finding the few that matter.

Overfitting risk
- The evidence-based prior shrinks pair effects that about 25 builds cannot
  support, but they stay rough estimates at that budget.
"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .common import RegressionPerformanceModel, StandardizedTarget


class BayesPairsPerformanceModel(RegressionPerformanceModel):
    native_uncertainty = True
    uncertainty_method = "Bayesian ridge predictive standard deviation (model-dependent)"

    def _make_regressor(self, seed, n_features):
        # Pairs of values of the same option are always zero: dropped as constant
        return make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False),
                             VarianceThreshold(),
                             StandardizedTarget(BayesianRidge(**self.estimator_params)))
