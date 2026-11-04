"""Seeded candidate generation and ranking shared by all performance models."""

from itertools import islice
from math import prod
from random import Random

import numpy as np

from .load import DEFAULT, HARDWARE_DETERMINED, valid


# Reproducibility and search budgets
SEARCH_SEED = 0
UCB_CANDIDATES = 2_000
FINAL_CANDIDATES = 10_000
SEARCH_STARTS = 5
DRAW_ATTEMPTS_PER_CANDIDATE = 10
MIN_GLOBAL_DRAW_ATTEMPTS = 1_000
SCORE_BATCH_SIZE = 4_096


class CandidateSearch:
    def __init__(self, values, hardware, application, rule_options=(), seed=SEARCH_SEED):
        self.values = {
            option: tuple(dict.fromkeys(map(str, choices)))
            for option, choices in values.items() if option not in HARDWARE_DETERMINED
        }
        if any(not choices for choices in self.values.values()):
            raise ValueError("each search option needs at least one allowed value")
        self.hardware = hardware
        self.application = application
        self.rules = {option: HARDWARE_DETERMINED[option](hardware) for option in rule_options}
        self.rng = Random(seed)

    def key(self, config):
        return tuple(str(config.get(option, DEFAULT)) for option in self.values)

    def complete(self, config):
        return {**dict(zip(self.values, self.key(config))), **self.rules}

    def _in_domain(self, config):
        return (
            all(value in choices for value, choices in zip(self.key(config), self.values.values()))
            and all(option in self.values or value == self.rules.get(option, DEFAULT)
                    for option, value in config.items())
        )

    def accepts(self, config):
        return self._in_domain(config) and valid(self.complete(config), self.hardware, self.application)

    def _draw(self, attempts):
        """Sample Cartesian-product indices without replacement or materialization."""
        # Lazy index shuffle
        remaining = prod(map(len, self.values.values()))
        swaps = {}
        for _ in range(min(attempts, remaining)):
            draw = self.rng.randrange(remaining)
            remaining -= 1
            index = swaps.get(draw, draw)
            swaps[draw] = swaps.get(remaining, remaining)
            config = {}
            for option, choices in reversed(self.values.items()):
                index, offset = divmod(index, len(choices))
                config[option] = choices[offset]
            yield config

    def _variants(self, anchors, attempts):
        # Small edits and occasional larger jumps
        for _ in range(attempts):
            anchor, alternatives = self.rng.choice(anchors)
            size = min(len(alternatives), self.rng.choice((1, 1, 2, 2, 3, 4)))
            changes = {option: self.rng.choice(alternatives[option])
                       for option in self.rng.sample(list(alternatives), size)}
            yield {**anchor, **changes}

    def candidates(self, limit, *, anchors=(), local_limit=0, excluded=()):
        """Return up to limit valid builds, filling unused local slots with global draws."""
        # Measured builds and duplicate proposals
        seen = {self.key(config) for config in excluded if self._in_domain(config)}
        candidates = []

        def collect(proposals, target):
            if len(candidates) >= target:
                return
            for config in proposals:
                key = self.key(config)
                if key not in seen:
                    seen.add(key)
                    if self.accepts(config):
                        candidates.append(self.complete(config))
                        if len(candidates) == target:
                            break

        # Local quota
        if local_limit and anchors and any(len(choices) > 1 for choices in self.values.values()):
            neighborhoods = [
                (anchor, {option: tuple(value for value in choices if value != current)
                          for option, choices, current in zip(
                              self.values, self.values.values(), self.key(anchor))
                          if len(choices) > 1})
                for anchor in anchors
            ]
            collect(self._variants(neighborhoods, local_limit * DRAW_ATTEMPTS_PER_CANDIDATE),
                    min(local_limit, limit))
        # Global draws and unused local slots
        attempts = max(MIN_GLOBAL_DRAW_ATTEMPTS, limit * DRAW_ATTEMPTS_PER_CANDIDATE)
        collect(self._draw(attempts), limit)
        return candidates

    def rank(self, model, builds, count, *, ucb=False):
        """Return (build, mean, spread) in descending score order."""
        # Batched scores and running top picks
        best = []
        builds = iter(builds)
        while batch := list(islice(builds, SCORE_BATCH_SIZE)):
            mean, spread = model.predict_performance(self.hardware, batch)
            scores = mean + spread if ucb else mean
            indices = np.argsort(-scores, kind="stable")[:count]
            best.extend((float(scores[i]), batch[i], float(mean[i]), float(spread[i])) for i in indices)
            best.sort(key=lambda item: item[0], reverse=True)
            del best[count:]
        return [(build, mean, spread) for _, build, mean, spread in best]

    def final(self, model, *, anchors=()):
        """Share one scoring budget between previous winners, fresh draws, and variants."""
        # Previous winners and fresh draws
        anchor_limit = min(2 * SEARCH_STARTS, FINAL_CANDIDATES // 2)
        anchors = list({self.key(config): self.complete(config) for config in anchors
                        if self.accepts(config)}.values())[:anchor_limit]
        builds = anchors + self.candidates(FINAL_CANDIDATES // 2, excluded=anchors)
        best = self.rank(model, builds, SEARCH_STARTS)
        if not best:
            raise ValueError("search found no valid configurations")
        # Local refinement
        remaining = FINAL_CANDIDATES - len(builds)
        starts = list({self.key(config): config
                       for config in anchors + [row[0] for row in best]}.values())
        variants = self.candidates(remaining, anchors=starts,
                                   local_limit=remaining, excluded=builds)
        refined = self.rank(model, variants, 1)
        return max(best[:1] + refined, key=lambda row: row[1])[:2]
