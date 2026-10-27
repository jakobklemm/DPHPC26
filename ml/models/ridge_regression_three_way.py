"""Bayesian ridge with build-option × build-option × hardware interactions.

trains on:
- build-option x hardware interaction
- build-option x build-option interaction
- build-option x build-option x hardware

Increase in parameters for training might lead to overfitting, as we have only
limited amount of data points available for training.
"""

import pandas as pd

from .ridge_regression_interactions import InteractionPerformanceModel


class ThreeWayPerformanceModel(InteractionPerformanceModel):
    """Let each learned option-pair effect vary with encoded hardware features.

    Retain individual options, hardware, option × hardware, and option × option
    terms. Add varying A × B × h products, with A and B from different options
    and h from hardware. Pure three-option interactions are not included.
    """

    def _encode(self, data, fit=False):
        X = super()._encode(data, fit=fit)
        if fit:
            self.hardware_option_pairs = [
                (a, b, h) for a, b in self.option_pairs for h in self.hw_columns
                if (X[a] * X[b] * X[h]).nunique() > 1
            ]

        # Reuse the training schema for batches, single predictions, and reloads.
        terms = {f"{a} x {b} x {h}": X[a] * X[b] * X[h]
                 for a, b, h in self.hardware_option_pairs}
        return pd.concat([X, pd.DataFrame(terms, index=X.index)], axis=1)
