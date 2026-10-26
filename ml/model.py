"""
Step 2: the surrogate model.

Why predict performance and not directly a build: a classifier (hardware ->
build) needs a correct answer per example, so it could only learn from the
fastest builds and would throw the slow ones away. Predicting performance,
every run is a valid example (the slow ones show what to avoid), and the
best build is then found by comparing the predictions.

A surrogate model replaces an expensive measurement with a cheap prediction.
Here it answers: "how fast would this build be on this machine?" without
building and benchmarking it.
    (hardware features, build configuration)  ->  predicted relative performance

Choosing a build for a machine: predict the performance of every candidate
build on that machine and take the one with the highest prediction.
"""

import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

# HARDWARE_DETERMINED and valid are defined per application in load_<app>.py
from .load import CFG, DEFAULT, HARDWARE_DETERMINED, HW, columns, strip, valid


# Bayesian ridge regression: a linear model that keeps the effects small unless the data clearly supports
# them (important with few runs), and that also says how unsure each prediction is (used by `suggest`)
def make_regressor():
    return BayesianRidge()


class PerformanceModel:

    # --- Training ----------------------------------------------------------------

    # train on all builds of an application (one row per build and machine, repeated runs already averaged)
    def _fit_options(self, table):
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

    def fit(self, table):
        self.hw_features = [strip(c) for c in columns(table, HW)]
        self._fit_options(table)

        # Input: hardware features + build options of every build. Output: the log of its relative performance,
        # so that effects multiply: -0.12 in log space is about 12 % slower, whatever the base performance.
        X = self._encode(table.rename(columns=strip)[self.hw_features + self.options], fit=True)
        y = np.log(table["relative"])
        self.regressor = make_regressor().fit(X, y)

        # How well the model fits
        self.r2 = self.regressor.score(X, y)
        self.noise = np.exp(np.sqrt(1 / self.regressor.alpha_)) - 1
        return self

    def _encode(self, data, fit=False):
        """
        Turn the inputs into numbers

        True/False become 1/0. Text (e.g. gpu_vendor, or an option like GGML_LTO=ON) becomes one 0/1 column per value, e.g. "GGML_LTO_ON"
        
        Interactions: every option column times every hardware column so that the effect of an option can differ between machines
        """
        X = pd.get_dummies(data.astype({c: float for c in data.select_dtypes(bool).columns}), dtype=float)
        if fit:
            reference = {f"{o}_{self.values[o][0]}" for o in self.options}
            option_values = {f"{o}_{v}" for o in self.options for v in self.values[o]}
            kept = [c for c in X.columns if X[c].nunique() > 1 and c not in reference]
            self.option_columns = [c for c in kept if c in option_values]
            self.hw_columns = [c for c in kept if c not in option_values]
        X = X.reindex(columns=self.option_columns + self.hw_columns, fill_value=0)
        for o in self.option_columns:
            for h in self.hw_columns:
                X[f"{o} x {h}"] = X[o] * X[h]
        if X.shape[1] == 0:
            # A baseline may have no varying options and no hardware features.
            X["constant"] = 1.0
        return X

    # --- Using the model ---------------------------------------------------------

    # predicted (relative) performance of each config on this machine & its uncertainty
    def predict_performance(self, hardware, configs):
        rows = pd.DataFrame([{**hardware, **{o: c.get(o, DEFAULT) for o in self.options}} for c in configs])
        X = self._encode(rows[self.hw_features + self.options])
        log_mean, log_std = self.regressor.predict(X, return_std=True)
        mean = np.exp(log_mean)
        return mean, mean * log_std

    # returns all combinations of the option values seen in training that are valid builds for the hardware
    def candidates(self, hardware):
        combinations = itertools.product(*(self.values[o] for o in self.options))
        configs = (dict(zip(self.options, values)) for values in combinations)
        return [c for c in configs if self._valid(c, hardware)]

    def _valid(self, config, hardware):
        # Complete output validation, including derived CUDA architecture
        return valid(self._add_rule_options(config, hardware), hardware, getattr(self, "application", None))

    # returns for a given hardware the config with the highest predicted performance, and the predicted performance too
    def predict(self, hardware):
        candidates = self.candidates(hardware)
        if not candidates:
            raise ValueError("no valid build configurations for this hardware")
        mean, _ = self.predict_performance(hardware, candidates)
        best = int(np.argmax(mean))
        return self._add_rule_options(candidates[best], hardware), float(mean[best])

    def _add_rule_options(self, config, hardware):
        return {**config, **{o: HARDWARE_DETERMINED[o](hardware) for o in self.rule_options}}

    # get the n most promising builds not yet measured on this machine, ranked by prediction + uncertainty
    def suggest(self, hardware, measured, n=5):
        measured = {tuple(c.get(o) for o in self.options) for c in measured}
        untested = [c for c in self.candidates(hardware) if tuple(c.values()) not in measured]
        if not untested:
            return []
        mean, spread = self.predict_performance(hardware, untested)
        order = np.argsort(-(mean + spread))[:n]
        return [(self._add_rule_options(untested[i], hardware), float(mean[i]), float(spread[i])) for i in order]

    # What each decision costs on this machine, only for printing.
    #   value        the option's value in the best build
    #   alternative  the other value
    #   also         other options that have to change with it to give a valid build
    #   cost         predicted change in performance, e.g. -0.03 = 3 % slower
    #   uncertainty  standard deviation of the cost
    def decisions(self, hardware):
        candidates = self.candidates(hardware)
        predicted = self.predict_performance(hardware, candidates)[0]
        best = candidates[int(np.argmax(predicted))]
        rows = []
        for option in self.options:
            for value in self.values[option]:
                if value == best[option]:
                    continue
                # the valid build with this value that differs least from the best one (then: predicted best)
                with_value = [(sum(c[o] != best[o] for o in self.options), -p, i)
                              for i, (c, p) in enumerate(zip(candidates, predicted)) if c[option] == value]
                if not with_value:
                    continue  # this value cannot be built on this machine
                variant = candidates[min(with_value)[2]]
                rows.append({"option": option, "value": best[option], "alternative": value,
                             "also": {o: variant[o] for o in self.options if o != option and variant[o] != best[o]},
                             "variant": variant})
        if not rows:
            return pd.DataFrame(rows)

        builds = pd.DataFrame([{**hardware, **c} for c in [best] + [r.pop("variant") for r in rows]])
        X = self._encode(builds[self.hw_features + self.options]).to_numpy()
        diff = X[1:] - X[0]
        log_change = diff @ self.regressor.coef_
        log_std = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.regressor.sigma_, diff))
        result = pd.DataFrame(rows)
        result["cost"] = np.exp(log_change) - 1
        result["uncertainty"] = np.exp(log_std) - 1
        return result

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
