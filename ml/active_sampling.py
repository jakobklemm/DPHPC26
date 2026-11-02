"""
Step 4: how few builds are enough? Active sampling, by default without pretraining.

A session on one machine and application starts with no model and a measurement
budget, e.g. 25 builds:
    ask        next build to measure: random at first, then the highest
               predicted performance + uncertainty (UCB)
    tell       add the measured performance and refit the model on all of them
    recommend  the best measured build

With a model pretrained on other machines (transfer learning), half of the random
start becomes that model's top builds, and the model learns only the error of the
pretrained model on this machine (see models/residual_correction.py).

Currently pool-based: builds are chosen from a fixed table of candidates, so
`simulate` can replay a session on builds that were already measured ("measuring"
is looking the result up) and compare the choices with the whole pool.

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

from .load import HW, columns, strip
from .models import MODEL_TYPES, ResidualCorrection
from .models.common import configs


class ActiveSampler:
    """One session over a pool of candidate builds; model_type None samples randomly.

    pretrained: a model fitted on other machines, corrected by model_type here.
    """

    def __init__(self, model_type, pool, initial=8, seed=0, pretrained=None):
        if initial < 2:
            raise ValueError("initial must be at least 2 builds")
        self.model_type = model_type
        self.pool = pool
        self.initial = initial
        self.candidates = configs(pool)
        self.hardware = {strip(c): pool[c].iloc[0] for c in columns(pool, HW)}
        self.order = np.random.default_rng(seed).permutation(len(pool))
        if pretrained is not None:
            source, _ = pretrained.predict_performance(self.hardware, self.candidates)
            top = np.argsort(-source)[:initial // 2]
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


def _session(name, pool, budget, initial, seed, pretrained=None):
    start = time.perf_counter()
    sampler = ActiveSampler(None if name == "random" else MODEL_TYPES[name], pool, initial, seed, pretrained)
    performance = pool["performance"].to_numpy()
    relative = pool["relative"].to_numpy()
    best = {}
    for n in range(1, budget + 1):
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


def simulate(pool, models, budget=25, initial=8, seeds=30, jobs=-1, pretrained=None):
    """Replay sessions on one machine's measured builds; one result row per session.

    The same seed gives every model the same random start, so that differences
    between models do not come from a lucky start. A pretrained model adds a
    corrected session per model, next to the sessions without pretraining.
    """
    if not initial <= budget < len(pool):
        raise ValueError(f"need initial <= budget < {len(pool)} measured builds")
    names = ["random"] + [name for name in models if name != "random"]
    variants = [(name, None) for name in names]
    if pretrained is not None:
        variants += [(name, pretrained) for name in names if name != "random"]
    sessions = Parallel(n_jobs=jobs)(delayed(_session)(name, pool, budget, initial, seed, source)
                                     for name, source in variants for seed in range(seeds))
    return pd.DataFrame(sessions)
