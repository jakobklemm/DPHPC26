"""Option-only and random baselines for build configuration selection."""

import heapq
import math

import numpy as np
import pandas as pd

from .load import CFG, HW, columns, valid
from .model import PerformanceModel


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


class NoHardwareModel(PerformanceModel):
    """Learn option effects only; hardware is used solely for build feasibility
    and to restore hardware-determined settings, never as a regression input.
    """

    def fit(self, table):
        return super().fit(table.drop(columns=columns(table, HW)))

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
            if valid(config, hardware):
                mean, _ = self.predict_performance(hardware, [config])
                return self._add_rule_options(config, hardware), float(mean[0])
            for dimension, values in enumerate(ranked):
                if indices[dimension] + 1 < len(values):
                    neighbor = indices[:dimension] + (indices[dimension] + 1,) + indices[dimension + 1:]
                    if neighbor not in seen:
                        seen.add(neighbor)
                        heapq.heappush(queue, (cost(neighbor), neighbor))
        raise ValueError("no valid build configurations for this hardware")


class RandomBaseline(PerformanceModel):
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
            if valid(configs[i], hardware):
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
            if valid(config, hardware):
                return self._add_rule_options(config, hardware), None
        raise ValueError("no valid build configurations for this hardware")
