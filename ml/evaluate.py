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
  baseline  the same for the build that was best on average on the other
            platforms, i.e. a choice that ignores the hardware
"""

import numpy as np
import pandas as pd

from .load import CFG, HW, columns, strip
from .model import PerformanceModel


def configs(table):
    cols = columns(table, CFG)
    return [{strip(c): row[c] for c in cols} for _, row in table[cols].iterrows()]


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


def baseline_pick(train, test, options):
    """Among the builds measured in `test`, the one with the best average on the training platforms """
    keys = [CFG + o for o in options]
    average = train.groupby(keys)["relative"].mean().rename("average").reset_index()
    candidates = test.merge(average, on=keys)
    if candidates.empty:
        return float("nan")
    return candidates.sort_values("average", ascending=False)["relative"].iloc[0]


def leave_one_platform_out(table):
    rows = []
    for platform in sorted(table["platform"].unique()):
        train = table[table["platform"] != platform]
        test = table[table["platform"] == platform]
        hardware = {strip(c): test[c].iloc[0] for c in columns(test, HW)}

        model = PerformanceModel().fit(train)
        predicted, _ = model.predict_performance(hardware, configs(test))
        measured = test["relative"].to_numpy()

        rows.append({
            "platform": platform,
            "runs": len(test),
            "error": float(np.mean(np.abs(predicted - measured))),
            "model": measured[int(np.argmax(predicted))],
            "baseline": baseline_pick(train, test, model.options),
        })
    return pd.DataFrame(rows)
