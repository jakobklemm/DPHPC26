"""
Step 3: how good are the predictions?

# Leave-one-build-out (works with a single machine): for every build, train on
# all other builds and predict the held-out one.
#
# Reported per build:
#   measured   its relative performance
#   predicted  what the model predicted without having seen it
#   error      |predicted - measured|
#   guess      error of the simplest guess, the average of the other builds;
#              the model is only useful if its error is smaller

Leave-one-platform-out (needs at least 2 machines): for every platform, train
on all other platforms and let the model predict the held-out one. 

Reported per platform:
  error     how far the predicted relative performance is from the measured
            one, on average over the measured builds
  model     measured relative performance of the build the model rates best
  baseline  selected build's performance using the best mean relative
            performance of complete configurations on training platforms
  no_hardware  selected build's performance using an option-only regression
  interactions  selected build's performance using option-pair ridge regression
  error_interactions  mean absolute prediction error of option-pair ridge
  no_hardware_interactions  option-pair regression without hardware inputs
  random       mean selected performance over repeated random choices
  random_std   standard deviation of those random choices

All methods choose from the same valid, measured builds, restricted to option
values seen in training. This evaluates ranking on measured builds, not the
performance of unmeasured combinations in the full candidate space.
"""

import numpy as np
import pandas as pd

from .load import HW, columns, strip
from .models.baseline_models import (NoHardwareInteractionModel, NoHardwareModel,
                                     RandomBaseline, baseline_pick)
from .models.ridge_regression import PerformanceModel
from .models.ridge_regression_interactions import InteractionPerformanceModel
from .models.common import configs


# def leave_one_build_out(table):
#     rows = []
#     for i in table.index:
#         # `relative` is recomputed without the held-out build: if it was the best one, the other
#         # builds' relative values would otherwise reveal how good it is
#         train = rescale(table.drop(index=i))
#         test = table.loc[i]
#         best = train.loc[train["platform"] == test["platform"], "performance"].max()
#         measured = test["performance"] / best
#
#         model = PerformanceModel().fit(train)
#         hardware = {strip(c): test[c] for c in columns(table, HW)}
#         predicted, _ = model.predict_performance(hardware, [{strip(c): test[c] for c in columns(table, CFG)}])
#         guess = train.loc[train["platform"] == test["platform"], "relative"].mean()
#
#         varying = [c for c in columns(table, CFG) if table[c].nunique() > 1]
#         rows.append({
#             "build": " ".join(f"{strip(c)}={test[c]}" for c in varying),
#             "measured": measured,
#             "predicted": predicted[0],
#             "error": abs(predicted[0] - measured),
#             "guess": abs(guess - measured),
#         })
#     return pd.DataFrame(rows)
#
#
# def rescale(table):
#     """Recompute `relative` (performance / best on the same platform) for a subset of the table."""
#     table = table.copy()
#     table["relative"] = table["performance"] / table.groupby("platform")["performance"].transform("max")
#     return table


def leave_one_platform_out(table, random_trials=1000, seed=0):
    if random_trials < 1:
        raise ValueError("random_trials must be at least 1")
    rows = []
    random = RandomBaseline(seed=seed)
    for platform in sorted(table["platform"].unique()):
        train = table[table["platform"] != platform]
        test = table[table["platform"] == platform]
        hardware = {strip(c): test[c].iloc[0] for c in columns(test, HW)}

        model = PerformanceModel().fit(train)
        no_hardware = NoHardwareModel().fit(train)
        interactions = InteractionPerformanceModel().fit(train)
        no_hardware_interactions = NoHardwareInteractionModel().fit(train)
        random.fit(train)
        eligible = [
            all(c.get(o) in model.values[o] for o in model.options) and model._valid(c, hardware)
            for c in configs(test)
        ]
        test = test.loc[eligible]
        if test.empty:
            rows.append({"platform": platform, "runs": 0, "error": np.nan,
                         "model": np.nan, "baseline": np.nan, "no_hardware": np.nan,
                         "interactions": np.nan, "error_interactions": np.nan,
                         "no_hardware_interactions": np.nan,
                         "random": np.nan, "random_std": np.nan})
            continue
        candidates = configs(test)
        predicted, _ = model.predict_performance(hardware, candidates)
        option_predicted, _ = no_hardware.predict_performance(hardware, candidates)
        pair_predicted, _ = interactions.predict_performance(hardware, candidates)
        pair_option_predicted, _ = no_hardware_interactions.predict_performance(hardware, candidates)
        measured = test["relative"].to_numpy()
        random_performance = np.array([
            measured[random.choose_index(hardware, candidates)] for _ in range(random_trials)
        ])

        rows.append({
            "platform": platform,
            "runs": len(test),
            "error": float(np.mean(np.abs(predicted - measured))),
            "model": measured[int(np.argmax(predicted))],
            "baseline": baseline_pick(train, test, model.options),
            "no_hardware": measured[int(np.argmax(option_predicted))],
            "interactions": measured[int(np.argmax(pair_predicted))],
            "error_interactions": float(np.mean(np.abs(pair_predicted - measured))),
            "no_hardware_interactions": measured[int(np.argmax(pair_option_predicted))],
            "random": float(random_performance.mean()),
            "random_std": float(random_performance.std()),
        })
    return pd.DataFrame(rows)
