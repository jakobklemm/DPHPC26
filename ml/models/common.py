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
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..load import CFG, DEFAULT, HARDWARE_DETERMINED, HW, columns, strip, valid


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


def category(value):
    """Keep absent values distinct from explicit CMake defaults and unknowns."""
    # Comparable scalar categories; distinct missing/default/unknown values
    return "__missing__" if pd.isna(value) else str(value)


def performance_space(log_prediction):
    """Exponentiate log predictions with finite numerical bounds, not a 1.0 cap."""
    # Bounded exponentiation; predictions above the measured best allowed
    return np.exp(np.clip(log_prediction, -300, 300))


class FeatureEncoder:
    """Training-only schema, imputation and optional scaling or native categories.

    Build options are always categorical. Constant training columns are omitted:
    their effects cannot be learned, even if prediction hardware changes them.
    Each bootstrap member owns its encoder, including its imputation statistics.
    """

    def __init__(self, scale=False, native=False):
        self.scale = scale
        self.native = native

    def fit(self, frame):
        # Continuous numeric hardware; categorical build settings
        numeric = [c for c in frame if c.startswith(HW)
                        and pd.api.types.is_numeric_dtype(frame[c])
                        and not pd.api.types.is_bool_dtype(frame[c])]
        categorical = [c for c in frame if c not in numeric]
        # Imputation statistics from this training sample only
        numbers = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.medians = numbers.median().fillna(0.0)
        numbers = numbers.fillna(self.medians)
        categories = frame[categorical].map(category)
        # Constant-input detection after normalization and imputation
        cleaned = pd.concat([numbers, categories], axis=1)
        self.columns = [c for c in frame if cleaned[c].nunique() > 1]
        self.numeric = [c for c in self.columns if c in numeric]
        self.categorical = [c for c in self.columns if c in categorical]
        self.medians = self.medians.reindex(self.numeric)
        # Optional model-specific numeric scaling
        if self.scale and self.numeric:
            self.scaler = StandardScaler().fit(numbers[self.numeric])
        # Native CatBoost categories; stable one-hot columns for sklearn
        # Unseen-category support within the fitted schema
        if self.categorical and not self.native:
            self.one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.one_hot.fit(categories[self.categorical])
        return self

    def transform(self, frame):
        # Reuse of fitted columns, imputation and scaling
        numbers = frame.reindex(columns=self.numeric).apply(pd.to_numeric, errors="coerce")
        numbers = numbers.replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        if self.scale and self.numeric:
            numbers = pd.DataFrame(self.scaler.transform(numbers), columns=self.numeric, index=frame.index)
        categories = frame.reindex(columns=self.categorical).map(category)
        # Named mixed-type table for CatBoost; numeric matrix otherwise
        if self.native:
            return pd.concat([numbers, categories], axis=1)
        pieces = [numbers.to_numpy(dtype=float)]
        if self.categorical:
            pieces.append(self.one_hot.transform(categories))
        return np.concatenate(pieces, axis=1)


class FittedRegressor:
    """One estimator and its independently fitted input transformation."""

    def __init__(self, encoder, regressor):
        self.encoder = encoder
        self.regressor = regressor

    def fit(self, frame, target):
        X = self.encoder.transform(frame)
        # Intercept fallback for constant inputs or targets
        self.constant = X.shape[1] == 0 or np.ptp(target) == 0
        if self.constant:
            self.regressor = DummyRegressor().fit(np.zeros((len(target), 1)), target)
            self.constant_std = float(np.std(target) / np.sqrt(len(target)))
        # Named categorical columns for CatBoost; numeric inputs otherwise
        elif self.encoder.native:
            self.regressor.fit(X, target, cat_features=self.encoder.categorical)
        else:
            self.regressor.fit(X, target)
        return self

    def predict(self, frame, return_std=False):
        # Fallback input shape and log-target standard error
        X = np.zeros((len(frame), 1)) if self.constant else self.encoder.transform(frame)
        if return_std and self.constant:
            return self.regressor.predict(X), np.full(len(frame), self.constant_std)
        # Native predictive uncertainty for the GP path
        if return_std:
            return self.regressor.predict(X, return_std=True)
        return np.asarray(self.regressor.predict(X)).reshape(-1)


class RegressionPerformanceModel(BaseModel):
    """Shared log-target surrogate with native or bootstrap uncertainty.

    Ensemble spread is an uncalibrated heuristic, not measurement noise or a
    confidence guarantee. Whole-platform bootstrap draws retain within-machine
    dependence; with one platform, rows are bootstrapped instead. Point estimates
    use the full-data fit, as in ridge, rather than averaging bootstrap models.
    """

    scale_features = False
    native_categories = False
    native_uncertainty = False
    uncertainty_method = "bootstrap prediction spread (uncalibrated)"

    def __init__(self, random_state=0, ensemble_size=8, **estimator_params):
        # Multiple bootstrap members and stored estimator overrides
        if not isinstance(ensemble_size, int) or isinstance(ensemble_size, bool) or ensemble_size < 2:
            raise ValueError("ensemble_size must be an integer of at least 2")
        self.random_state = random_state
        self.ensemble_size = ensemble_size
        self.estimator_params = estimator_params

    def _make_regressor(self, seed, n_features):
        raise NotImplementedError

    def _fit_member(self, frame, target, seed):
        # Independent encoder and statistics per fit
        # Encoded feature count for dimension-dependent kernel defaults
        encoder = FeatureEncoder(self.scale_features, self.native_categories)
        n_features = encoder.fit(frame).transform(frame).shape[1]
        estimator = self._make_regressor(seed, max(1, n_features))
        return FittedRegressor(encoder, estimator).fit(frame, target)

    def fit(self, table):
        # Target validation before log transformation and fitting
        if table.empty:
            raise ValueError("cannot train a performance model on an empty table")
        target = table["relative"].to_numpy(dtype=float)
        if not np.all(np.isfinite(target) & (target > 0)):
            raise ValueError("relative performance must be positive and finite")
        # Build-setting normalization on a copy of the benchmark table
        table = table.copy()
        for c in columns(table, CFG):
            table[c] = table[c].map(category)
        # Shared option domains and hardware-derived rules
        self._fit_options(table)
        self.hw_features = columns(table, HW)
        self.feature_columns = self.hw_features + [CFG + o for o in self.options]
        frame = table[self.feature_columns]
        # Additive log-space representation of multiplicative performance effects
        target = np.log(target)
        rng = np.random.default_rng(self.random_state)
        # Full-data point estimates; training-only R² diagnostic
        self.regressor = self._fit_member(frame, target, self.random_state)
        fitted = self.regressor.predict(frame)
        self.r2 = float(r2_score(target, fitted)) if len(target) > 1 else float("nan")
        self.ensemble = []
        # Native GP uncertainty; bootstrap spread for other models
        if not self.native_uncertainty:
            groups = ([np.flatnonzero(table["platform"].to_numpy() == p)
                       for p in table["platform"].unique()] if "platform" in table else [])
            for _ in range(self.ensemble_size):
                if len(groups) > 1:
                    # Whole-machine resampling to preserve within-machine dependence
                    indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                else:
                    # Row resampling for single-machine data
                    indices = rng.integers(len(table), size=len(table))
                # Derived member seeds and independent sample fits
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                self.ensemble.append(self._fit_member(frame.iloc[indices], target[indices], seed))
        return self

    def _prediction_frame(self, hardware, configurations):
        # Target hardware paired with each build in the training schema
        # CMake defaults for omitted settings; missing values for absent hardware
        rows = [{**{c: hardware.get(strip(c), np.nan) for c in self.hw_features},
                 **{CFG + o: category(config.get(o, DEFAULT)) for o in self.options}}
                for config in configurations]
        return pd.DataFrame(rows, columns=self.feature_columns)

    def predict_performance(self, hardware, configs):
        # Matching empty outputs for zero candidate builds
        if not configs:
            return np.empty(0), np.empty(0)
        frame = self._prediction_frame(hardware, configs)
        if self.native_uncertainty:
            # GP log-space conversion using the ridge uncertainty approximation
            log_prediction, log_std = self.regressor.predict(frame, return_std=True)
            prediction = performance_space(log_prediction)
            return prediction, prediction * log_std
        # Full-data point estimate; bootstrap spread after exponentiation
        # Uncalibrated spread for exploration
        prediction = performance_space(self.regressor.predict(frame))
        draws = np.array([performance_space(member.predict(frame)) for member in self.ensemble])
        return prediction, np.std(draws, axis=0)

    def suggest(self, hardware, measured, n=5):
        # Measured-flag normalization for consistent build exclusion
        # Shared validity checks and performance-plus-spread ranking
        measured = [{o: category(c.get(o, DEFAULT)) for o in self.options} for c in measured]
        return super().suggest(hardware, measured, n)

