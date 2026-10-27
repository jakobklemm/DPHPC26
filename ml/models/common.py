"""Shared model behavior independent of the regression algorithm.

New performance surrogates subclass BaseModel, implement fit(table) and
predict_performance(hardware, configs), and reuse candidate validation,
selection, suggestions, and persistence. predict_performance returns parallel
arrays of relative performance predictions and uncertainty estimates.
Models with a different selection strategy can override predict/suggest.
"""

import itertools

import joblib
import numpy as np

from ..load import CFG, HARDWARE_DETERMINED, HW, columns, strip, valid


def configs(table):
    """Extract build configurations from a benchmark table as plain dictionaries."""
    cols = columns(table, CFG)
    return [{strip(c): row[c] for c in cols} for _, row in table[cols].iterrows()]


class NoHardwareMixin:
    """Exclude hardware from fitting while retaining it for build validity."""

    def fit(self, table):
        return super().fit(table.drop(columns=columns(table, HW)))


class BaseModel:
    """Common option domains, build validity, selection, and model storage."""

    def fit(self, table):
        """Fit the model, calling _fit_options(table) to record option domains."""
        raise NotImplementedError("models must implement fit")

    def predict_performance(self, hardware, configs):
        """Return relative performance predictions and uncertainty for each build."""
        raise NotImplementedError("this model does not predict performance")

    def _fit_options(self, table):
        """Record one application and the observed domains of its build options."""
        applications = table["application"].unique() if "application" in table else []
        if len(applications) > 1:
            raise ValueError("train one application at a time")
        self.application = applications[0] if len(applications) else None
        options = [strip(c) for c in columns(table, CFG)]
        self.rule_options = [o for o in options if o in HARDWARE_DETERMINED]
        self.options = [o for o in options if o not in HARDWARE_DETERMINED]

        # Every value seen for each option. All combinations of them are the
        # candidate builds that `predict` and `suggest` choose from.
        self.values = {o: sorted(table[CFG + o].unique()) for o in self.options}

    def candidates(self, hardware):
        """Combine observed option values and retain valid builds for this machine."""
        combinations = itertools.product(*(self.values[o] for o in self.options))
        configs = (dict(zip(self.options, values)) for values in combinations)
        return [c for c in configs if self._valid(c, hardware)]

    def _valid(self, config, hardware):
        # Complete output validation, including derived CUDA architecture
        return valid(self._add_rule_options(config, hardware), hardware, getattr(self, "application", None))

    def predict(self, hardware):
        """Return the valid build with highest predicted performance and its score."""
        candidates = self.candidates(hardware)
        if not candidates:
            raise ValueError("no valid build configurations for this hardware")
        mean, _ = self.predict_performance(hardware, candidates)
        best = int(np.argmax(mean))
        return self._add_rule_options(candidates[best], hardware), float(mean[best])

    def _add_rule_options(self, config, hardware):
        return {**config, **{o: HARDWARE_DETERMINED[o](hardware) for o in self.rule_options}}

    def suggest(self, hardware, measured, n=5):
        """Rank unmeasured valid builds by prediction plus uncertainty."""
        measured = {tuple(c.get(o) for o in self.options) for c in measured}
        untested = [c for c in self.candidates(hardware) if tuple(c.values()) not in measured]
        if not untested:
            return []
        mean, spread = self.predict_performance(hardware, untested)
        order = np.argsort(-(mean + spread))[:n]
        return [(self._add_rule_options(untested[i], hardware), float(mean[i]), float(spread[i])) for i in order]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

