"""
Step 4: how few builds are enough? Active sampling, by default without pretraining.

A session on one machine and application starts with no model and a measurement
number of steps, e.g. 25 builds:
    ask        next build to measure: random at first, then the highest
               predicted performance + uncertainty (UCB)
    tell       add the measured performance and refit the model on all of them
    recommend  the best measured build

With a model pretrained on other machines (transfer learning), the start builds
are that model's top builds instead of random ones, and the model learns only the
error of the pretrained model on this machine (see models/residual_correction.py).

`LiveActiveSampler` searches a caller-supplied flag space. `simulate` replays
sessions on previously measured builds and compares choices with that pool.

Reported per model, averaged over repeated sessions with different random starts:
  best@n          relative performance of the best build among the first n measured
                  (1.0 is the best build of the pool)
  model_pick      relative performance of the build the final model rates best
  rank_correlation  Spearman correlation of predicted and measured performance
                  on the builds the session did not measure
  seconds         time per session
`random` measures random builds only, the baseline any model has to beat.
`corrected-<model>` uses <model> to correct the pretrained model.
"""

import time
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .load import APPS, CFG, HW, columns, strip
from .models import MODEL_TYPES, ResidualCorrection
from .models.common import configs
from .search import CandidateSearch, SEARCH_SEED, SEARCH_STARTS, UCB_CANDIDATES


# Start builds before the local model: a fit on a single build predicts the same for every build
MIN_INITIAL = 2
DEFAULT_INITIAL = 2


class LiveActiveSampler:
    """Ask for build flags, benchmark them on the target, then tell the sampler the result.

    values maps flag names to their allowed values independently of the measured
    builds. Pass a model constructor from MODEL_TYPES, or None for random search.
    Start: `initial` builds, the pretrained model's top picks (random without one),
    then the local model. source names what chose the last build: "random",
    "pretrained" or "model".
    """

    def __init__(self, model_type, values, hardware, application, *, initial=DEFAULT_INITIAL,
                 seed=SEARCH_SEED, pretrained=None):
        if initial < MIN_INITIAL:
            raise ValueError(f"initial must be at least {MIN_INITIAL} builds")
        if model_type is None and pretrained is not None:
            raise ValueError("transfer correction requires a local model")
        self.initial = initial
        self.search = CandidateSearch(values, hardware, application,
                                      APPS[application].HARDWARE_DETERMINED, seed)
        self.model_type = model_type
        self.model = None
        self.measured = {}
        self.failed = {}  # builds that did not build or run; never proposed again
        self.promising = []
        self.warm = []
        self.pretrained = pretrained
        self.source = None
        if pretrained is not None:
            # Historical warm start
            self.model_type = partial(ResidualCorrection, pretrained, model_type, hardware)
            builds = self.search.candidates(UCB_CANDIDATES)
            self.warm = [row[0] for row in self.search.rank(pretrained, builds, initial)]

    def ask(self):
        """Return flags for the next build to benchmark."""
        measured = [config for config, _ in self.measured.values()] + list(self.failed.values())
        if self.model is None:
            builds = [config for config in self.warm
                      if self.search.key(config) not in self.measured and self.search.key(config) not in self.failed]
            self.source = "pretrained"
            if not builds and self.pretrained is not None:
                # Top picks used up by failed builds: the pretrained model chooses again
                builds = self.search.candidates(UCB_CANDIDATES, excluded=measured)
                builds = [row[0] for row in self.search.rank(self.pretrained, builds, 1)]
            elif not builds:
                builds = self.search.candidates(1, excluded=measured)
                self.source = "random"
        else:
            self.source = "model"
            # Mixed batch and UCB ranking
            builds = self.search.candidates(
                UCB_CANDIDATES, anchors=self._anchors(), local_limit=UCB_CANDIDATES // 2,
                excluded=measured)
            builds = [row[0] for row in self.search.rank(self.model, builds, SEARCH_STARTS, ucb=True)]
            self.promising = builds
        if not builds:
            raise ValueError("search found no valid unmeasured configurations")
        if not self.search.accepts(builds[0]):
            raise ValueError("sampling returned a disallowed build configuration")
        return builds[0].copy()

    def tell(self, config, performance):
        """Record one successful benchmark and refit on all local results."""
        key = self.search.key(config)
        if key in self.measured or key in self.failed or not self.search.accepts(config):
            raise ValueError("expected a valid, unmeasured configuration")
        if not np.isfinite(performance) or performance <= 0:
            raise ValueError("performance must be positive and finite")
        config = self.search.complete(config)
        if self.model_type is not None and len(self.measured) + 1 >= self.initial:
            # Full local refit
            measurements = [*self.measured.values(), (config, performance)]
            table = pd.DataFrame(
                {"application": self.search.application,
                 **{HW + name: value for name, value in self.search.hardware.items()},
                 **{CFG + name: value for name, value in build.items()},
                 "performance": score}
                for build, score in measurements)
            self.model = self.model_type().fit(table.assign(
                relative=table.performance / table.performance.max()))
        self.measured[key] = config, performance

    def fail(self, config):
        """Record a build that could not be built or benchmarked; ask() skips it."""
        key = self.search.key(config)
        if key in self.measured:
            raise ValueError("build already has a measured result")
        self.failed[key] = self.search.complete(config)

    def _anchors(self):
        best_measured = sorted(self.measured.values(), key=lambda row: row[1], reverse=True)
        return [config for config, _ in best_measured[:SEARCH_STARTS]] + self.promising

    def recommend(self):
        """Return the fastest measured build."""
        return max(self.measured.values(), key=lambda row: row[1])[0].copy()

    def predict(self):
        """Return the final model's top configuration and predicted score."""
        if self.model is None:
            raise ValueError("final prediction requires a fitted model")
        return self.search.final(self.model, anchors=self._anchors())


class ActiveSampler:
    """One session over a pool of candidate builds; model_type None samples randomly.

    pretrained: a model fitted on other machines, corrected by model_type here.
    """

    def __init__(self, model_type, pool, initial=DEFAULT_INITIAL, seed=SEARCH_SEED, pretrained=None):
        if initial < MIN_INITIAL:
            raise ValueError(f"initial must be at least {MIN_INITIAL} builds")
        self.model_type = model_type
        self.pool = pool
        self.initial = initial
        self.candidates = configs(pool)
        self.hardware = {strip(c): pool[c].iloc[0] for c in columns(pool, HW)}
        self.order = np.random.default_rng(seed).permutation(len(pool))
        if pretrained is not None:
            source, _ = pretrained.predict_performance(self.hardware, self.candidates)
            top = np.argsort(-source)[:initial]
            self.order = np.concatenate([top, self.order[~np.isin(self.order, top)]])
            self.model_type = partial(ResidualCorrection, pretrained, model_type, self.hardware)
        self.measured = {}  # pool row -> performance, in measurement order
        self.model = None

    def predicted(self):
        return self.model.predict_performance(self.hardware, self.candidates)

    def ask(self):
        """Pool row of the next build to measure."""
        if self.model is None:
            return int(next(row for row in self.order if row not in self.measured))
        mean, spread = self.predicted()
        score = mean + spread
        score[list(self.measured)] = -np.inf
        return int(np.argmax(score))

    def tell(self, row, performance):
        """Add one measurement and refit on everything measured so far."""
        self.measured[row] = performance
        if self.model_type is not None and len(self.measured) >= self.initial:
            # Relative to the best build so far: the session knows nothing else
            measured = np.array(list(self.measured.values()))
            table = self.pool.iloc[list(self.measured)].assign(
                performance=measured, relative=measured / measured.max())
            self.model = self.model_type().fit(table)

    def recommend(self):
        """Pool row of the best measured build."""
        return max(self.measured, key=self.measured.get)


def _session(name, pool, steps, initial, seed, pretrained=None):
    start = time.perf_counter()
    sampler = ActiveSampler(None if name == "random" else MODEL_TYPES[name], pool, initial, seed, pretrained)
    performance = pool["performance"].to_numpy()
    relative = pool["relative"].to_numpy()
    best = {}
    for n in range(1, steps + 1):
        row = sampler.ask()
        sampler.tell(row, performance[row])
        best[f"best@{n}"] = relative[sampler.recommend()]
    result = {"model": name if pretrained is None else f"corrected-{name}", "seed": seed, **best}
    if sampler.model is not None:
        mean, _ = sampler.predicted()
        rest = np.setdiff1d(np.arange(len(pool)), list(sampler.measured))
        result["model_pick"] = relative[int(np.argmax(mean))]
        result["rank_correlation"] = pd.Series(mean[rest]).rank().corr(pd.Series(relative[rest]).rank())
    return {**result, "seconds": time.perf_counter() - start}


def simulate(pool, models, steps=25, initial=DEFAULT_INITIAL, seeds=30, jobs=-1, pretrained=None):
    """Replay sessions on one machine's measured builds; one result row per session.

    The same seed gives every model the same random start, so that differences
    between models do not come from a lucky start. A pretrained model adds a
    corrected session per model, next to the sessions without pretraining.
    """
    if not initial <= steps < len(pool):
        raise ValueError(f"need initial <= steps < {len(pool)} measured builds")
    names = ["random"] + [name for name in models if name != "random"]
    variants = [(name, None) for name in names]
    if pretrained is not None:
        variants += [(name, pretrained) for name in names if name != "random"]
    sessions = Parallel(n_jobs=jobs)(delayed(_session)(name, pool, steps, initial, seed, source)
                                     for name, source in variants
                                     for seed in range(SEARCH_SEED, SEARCH_SEED + seeds))
    return pd.DataFrame(sessions)
