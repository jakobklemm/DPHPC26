"""
Baselines: does learning hardware-specific option effects improve build choices?

Compare the hardware-aware models with simpler selection methods:

  baseline     best-on-average complete configuration (baseline_pick)
               Rank configurations by mean relative performance on training
               platforms, then choose the highest-ranked one measured on the
               held-out platform. Return its measured relative performance,
               or NaN if no configuration matches. Hardware-determined options
               are excluded from matching by the caller.

  no_hardware  option-only regression (NoHardwareModel)
               Learn individual option effects with Bayesian ridge regression,
               without hardware features or hardware-option interactions.
               Search combinations in predicted score order and return the
               first valid build with its predicted relative performance.

  random       uniform random selection (RandomBaseline)
               Record option values seen in training without learning from
               performance. Try combinations in random order without replacement
               until a valid build is found. Return the build and None, since
               there is no performance prediction. A seed makes choices repeatable.

  no_hardware_interactions (NoHardwareInteractionModel)
               Learn option effects and option pairs without hardware inputs.
               Score all valid combinations: independent option ranking is no
               longer correct when flags interact.

  no-hardware-{random-forest,catboost,gaussian-process,kernel-ridge,
               support-vector,small-mlp}
               Option-only counterparts of the nonlinear surrogates. Reuse
               each estimator's defaults and uncertainty method, excluding
               hardware inputs through NoHardwareMixin. Score all valid builds.

For prediction, no_hardware and random combine option values seen in training,
including combinations never measured together. Both search lazily, use hardware
only for validity checks and hardware-determined settings, and raise ValueError
if no valid build exists.

Evaluation instead restricts all methods to the same valid, measured candidates.
It reports the selected build's measured performance relative to the platform's
best measured build (1.0 is best), averaging repeated choices for random.
"""

import heapq
import math

import numpy as np
import pandas as pd

from ..load import CFG
from .common import BaseModel, NoHardwareMixin
from .ridge_regression import PerformanceModel
from .ridge_regression_interactions import InteractionPerformanceModel
from .random_forest import RandomForestPerformanceModel
from .catboost import CatBoostPerformanceModel
from .gaussian_process import GaussianProcessPerformanceModel
from .kernel_ridge import KernelRidgePerformanceModel
from .support_vector import SupportVectorPerformanceModel
from .small_mlp import SmallMLPPerformanceModel


def baseline_pick(train, test, options):
    """Return held-out relative performance of the best-on-average build.

    Rank complete option combinations by their mean relative performance on
    training platforms only. Candidates must also have a measured test result.
    Hardware-determined options are excluded by the caller via `options`.
    """
    if test.empty:
        return float("nan")
    keys = [CFG + o for o in options]
    if not keys:
        # All builds have the same empty configuration; choose the first tie.
        return float(test["relative"].iloc[0])
    average = train.groupby(keys)["relative"].mean().rename("average").reset_index()
    candidates = test.merge(average, on=keys)
    if candidates.empty:
        return float("nan")
    return float(candidates.sort_values("average", ascending=False, kind="stable")["relative"].iloc[0])


class NoHardwareModel(NoHardwareMixin, PerformanceModel):
    """Learn option effects only; hardware is used solely for build feasibility
    and to restore hardware-determined settings, never as a regression input.
    """

    def predict(self, hardware):
        # With no hardware interactions, option effects add in log space.
        # Search combinations from highest to lowest score, stopping at the
        # first valid one instead of materializing the full Cartesian product.
        reference = {o: self.values[o][0] for o in self.options}
        variants = [{**reference, o: v} for o in self.options for v in self.values[o]]
        data = pd.DataFrame([reference] + variants, columns=self.options)
        scores = iter(self.regressor.predict(self._encode(data))[1:])
        ranked = [sorted([(float(next(scores)), v) for v in self.values[o]],
                         key=lambda pair: pair[0], reverse=True) for o in self.options]
        start = (0,) * len(self.options)

        def cost(indices):
            return -sum(values[i][0] for values, i in zip(ranked, indices))

        queue = [(cost(start), start)]
        seen = {start}
        while queue:
            _, indices = heapq.heappop(queue)
            config = {o: values[i][1] for o, values, i in zip(self.options, ranked, indices)}
            if self._valid(config, hardware):
                mean, _ = self.predict_performance(hardware, [config])
                return self._add_rule_options(config, hardware), float(mean[0])
            for dimension, values in enumerate(ranked):
                if indices[dimension] + 1 < len(values):
                    neighbor = indices[:dimension] + (indices[dimension] + 1,) + indices[dimension + 1:]
                    if neighbor not in seen:
                        seen.add(neighbor)
                        heapq.heappush(queue, (cost(neighbor), neighbor))
        raise ValueError("no valid build configurations for this hardware")


class NoHardwareInteractionModel(NoHardwareMixin, InteractionPerformanceModel):
    """Option-pair ridge baseline; hardware is used only for build validity.

    Inherit exhaustive selection from BaseModel rather than the additive-only
    search in NoHardwareModel, because the best choice depends on flag pairs.
    """


class NoHardwareRandomForestModel(NoHardwareMixin, RandomForestPerformanceModel):
    """Option-only random forest with the hardware-aware model's defaults."""


class NoHardwareCatBoostModel(NoHardwareMixin, CatBoostPerformanceModel):
    """Option-only CatBoost with native categorical build settings."""


class NoHardwareGaussianProcessModel(NoHardwareMixin, GaussianProcessPerformanceModel):
    """Option-only Gaussian process with native predictive uncertainty."""


class NoHardwareKernelRidgeModel(NoHardwareMixin, KernelRidgePerformanceModel):
    """Option-only kernel ridge with the hardware-aware model's defaults."""


class NoHardwareSupportVectorModel(NoHardwareMixin, SupportVectorPerformanceModel):
    """Option-only support vector regression with the same estimator defaults."""


class NoHardwareSmallMLPModel(NoHardwareMixin, SmallMLPPerformanceModel):
    """Option-only small MLP with the hardware-aware model's defaults."""


class RandomBaseline(BaseModel):
    """Uniform random configuration, retrying invalid combinations.

    Fit records option domains only. No performance values are learned.
    predict returns (configuration, None), since this baseline has no score.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def fit(self, table):
        self.hw_features = []
        self._fit_options(table)
        return self

    def choose_index(self, hardware, configs):
        # Random order without replacement gives a uniform valid choice and
        # terminates even when every candidate is invalid.
        for i in self.rng.permutation(len(configs)):
            if self._valid(configs[i], hardware):
                return int(i)
        raise ValueError("no valid build configurations for this hardware")

    def predict(self, hardware):
        # Lazy Fisher-Yates shuffle of combination indices. Memory grows only
        # with rejected draws, not with the potentially huge candidate space.
        sizes = [len(self.values[o]) for o in self.options]
        remaining = math.prod(sizes)
        swaps = {}
        while remaining:
            draw = int(self.rng.integers(remaining))
            index = swaps.get(draw, draw)
            remaining -= 1
            swaps[draw] = swaps.get(remaining, remaining)
            config = {}
            for option, size in reversed(list(zip(self.options, sizes))):
                index, value = divmod(index, size)
                config[option] = self.values[option][value]
            config = {o: config[o] for o in self.options}
            if self._valid(config, hardware):
                return self._add_rule_options(config, hardware), None
        raise ValueError("no valid build configurations for this hardware")
