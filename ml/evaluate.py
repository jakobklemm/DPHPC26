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
  three_way  option-pair × hardware ridge selection score
  error_three_way  mean absolute prediction error of three-way ridge
  random       mean selected performance over repeated random choices
  random_std   standard deviation of those random choices

All methods choose from the same valid, measured builds, restricted to option
values seen in training. This evaluates ranking on measured builds, not the
performance of unmeasured combinations in the full candidate space.
"""

import numpy as np
import pandas as pd

from .load import HW, columns, strip
from .models import MODEL_TYPES
from .models.baseline_models import baseline_pick
from .models.common import configs


# Existing ridge/baseline column names for result compatibility
# Automatic score/error column pairs for new models
EVALUATION_NAMES = {
    "hardware": ("model", "error"),
    "hardware-interactions": ("interactions", "error_interactions"),
    "no-hardware": ("no_hardware", None),
    "no-hardware-interactions": ("no_hardware_interactions", None),
    "hardware-three-way": ("three_way", "error_three_way"),
}


def leave_one_platform_out(table, random_trials=1000, seed=0):
    """Compare all registered models on identical eligible measured candidates.

    New model score/error columns use underscores in place of CLI hyphens.
    Each model fits preprocessing only on the outer training platforms. There is
    no automatic hyperparameter tuning and no held-out-platform fit information.
    """
    # Positive random-trial count and separate training platform
    if random_trials < 1:
        raise ValueError("random_trials must be at least 1")
    if table["platform"].nunique() < 2:
        raise ValueError("leave-one-platform-out evaluation needs at least two platforms")
    # Shared CLI model registry
    # Separate random baseline without performance predictions
    names = {}
    for name in MODEL_TYPES:
        if name != "random":
            score = name.replace("-", "_")
            names[name] = EVALUATION_NAMES.get(name, (score, "error_" + score))
    rows = []
    # One seed for reproducible random choices across folds
    random = MODEL_TYPES["random"](seed=seed)
    for platform in sorted(table["platform"].unique()):
        # Whole-machine holdout; training on other machines only
        train = table[table["platform"] != platform]
        test = table[table["platform"] == platform]
        hardware = {strip(c): test[c].iloc[0] for c in columns(test, HW)}
        # Original eligibility rules shared by all models
        # Training-observed option values and valid builds only
        # Single ridge fit reused for scoring
        reference = MODEL_TYPES["hardware"]().fit(train)
        eligible = [all(c.get(o) in reference.values[o] for o in reference.options)
                    and reference._valid(c, hardware) for c in configs(test)]
        test = test.loc[eligible]
        # NaN scores for platforms without eligible builds
        row = {"platform": platform, "runs": len(test), "baseline": np.nan,
               "random": np.nan, "random_std": np.nan}
        for score, error in names.values():
            row[score] = np.nan
            if error:
                row[error] = np.nan
        if not test.empty:
            # Identical measured candidates across models
            candidates = configs(test)
            measured = test["relative"].to_numpy()
            for name, (score, error) in names.items():
                model = reference if name == "hardware" else MODEL_TYPES[name]().fit(train)
                # Predicted best build; measured performance as selection score
                predicted, _ = model.predict_performance(hardware, candidates)
                row[score] = float(measured[int(np.argmax(predicted))])
                if error:
                    # Mean absolute error in relative-performance space
                    row[error] = float(np.mean(np.abs(predicted - measured)))
            # Existing best-on-average and repeated uniform-random baselines
            row["baseline"] = baseline_pick(train, test, reference.options)
            random.fit(train)
            draws = np.array([measured[random.choose_index(hardware, candidates)]
                              for _ in range(random_trials)])
            row["random"] = float(draws.mean())
            row["random_std"] = float(draws.std())
        rows.append(row)
    # Existing column order followed by new model scores/errors
    legacy = ["platform", "runs", "error", "model", "baseline", "no_hardware",
              "interactions", "error_interactions", "no_hardware_interactions",
              "three_way", "error_three_way", "random", "random_std"]
    result = pd.DataFrame(rows)
    return result[legacy + [c for c in result if c not in legacy]]
