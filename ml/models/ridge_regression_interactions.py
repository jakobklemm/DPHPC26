"""Bayesian ridge with pairwise build-option effects in addition to hardware terms.

trains on:
- build-option x hardware interaction
- build-option x build-option interaction
"""

import itertools

import pandas as pd

from .ridge_regression import PerformanceModel


class InteractionPerformanceModel(PerformanceModel):
    """Learn option pairs shared across machines; retain option × hardware effects.

    Each pair multiplies two encoded values from different build options. Pairs
    that never vary in training are omitted. No option-pair × hardware or
    higher-order terms are added. Fitting, uncertainty, and decision reports
    reuse the original ridge implementation.
    """

    def _encode(self, data, fit=False):
        # Build settings are discrete choices, including numeric/boolean values.
        # Treat them categorically so they cannot be mistaken for hardware inputs.
        data = data.copy()
        for option in self.options:
            data[option] = data[option].astype(str)
        X = super()._encode(data, fit=fit)

        if fit:
            owner = {f"{option}_{value}": option
                     for option in self.options for value in self.values[option]}
            self.option_pairs = [
                (a, b) for a, b in itertools.combinations(self.option_columns, 2)
                if owner[a] != owner[b] and (X[a] * X[b]).nunique() > 1
            ]

        # Save and reuse the training pairs, even when predicting just one build.
        pairs = {f"{a} x {b}": X[a] * X[b] for a, b in self.option_pairs}
        return pd.concat([X, pd.DataFrame(pairs, index=X.index)], axis=1)
