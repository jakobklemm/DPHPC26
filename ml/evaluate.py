"""
Sampling evaluation: evaluate_sampling writes per-round traces for all model pairs.
The legacy leave_one_platform_out evaluation below remains available via
`python -m ml.main evaluate --mode platform`.

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


# Local models in evaluate_sampling only see one machine, so their hardware
# inputs are constant: these duplicate their no-hardware-* twins (three-way
# reduces to option pairs). Excluded from the default local list; --model can
# still request them.
SINGLE_MACHINE_DUPLICATES = {
    "hardware", "hardware-interactions", "hardware-three-way", "random-forest", "catboost",
    "gaussian-process", "kernel-ridge", "support-vector", "small-mlp",
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


def evaluate_sampling(table, output, *, models=None, historical=None, steps=25,
                      initial=2, seeds=3, jobs=-1, seed=0):
    """Evaluate every historical/local pair, holding out each target platform.

    Random is a cold-only baseline because it cannot fit residual corrections.
    Default local models skip SINGLE_MACHINE_DUPLICATES (hardware variants that
    match their no-hardware twins on a single machine); pass models to override.
    Every policy, cold or historical, runs once per seed in seed..seed+seeds-1.
    A seed sets the random start builds, the historical model's random_state and
    the local model's random_state, so seeds also vary stochastic fits (forests,
    MLPs, bootstraps). Deterministic models (ridge) ignore random_state.
    Output is streamed per completed session; use a fresh directory per run.
    """
    from pathlib import Path
    from joblib import Parallel, delayed
    from .active_sampling import _session, seeded
    from .models.common import option_values, unseen_count
    from .evaluation_trace import save_tables, validate_budget
    from .load import app_options

    validate_budget(steps, initial)
    if seeds < 1 or jobs == 0:
        raise ValueError('require seeds >= 1 and jobs != 0')
    models = models or [name for name in MODEL_TYPES if name not in SINGLE_MACHINE_DUPLICATES]
    historical = historical or ['none'] + [name for name in MODEL_TYPES if name != 'random']
    output = Path(output)
    if output.exists() and any(output.glob('*.csv')):
        raise ValueError(f'evaluation CSVs already exist in {output}; choose a fresh --output directory')
    # Validate every fold before writing partial results.
    for app, runs in table.groupby('application'):
        if any(name != 'none' for name in historical) and runs.platform.nunique() < 2:
            raise ValueError(f'{app}: historical evaluation requires at least two platforms')
        if any(len(pool) < steps for _, pool in runs.groupby('platform')):
            raise ValueError(f'{app}: every pool must contain at least {steps} builds after workload filtering')
    output.mkdir(parents=True, exist_ok=True)
    session_seeds = range(seed, seed + seeds)
    for app, runs in table.groupby('application'):
        for platform, pool in runs.groupby('platform'):
            pool = app_options(pool).reset_index(drop=True)
            train = app_options(runs[runs.platform != platform])
            # Same training data for every historical model: one count per target
            unseen = unseen_count(option_values(train), option_values(pool))
            for source_name in historical:
                # One historical fit per seed (a single fit for deterministic models)
                sources = dict.fromkeys(session_seeds)
                if source_name != 'none':
                    fits = {}
                    for session_seed in session_seeds:
                        model_type = seeded(source_name, session_seed)
                        stochastic = model_type is not MODEL_TYPES[source_name]
                        key = session_seed if stochastic else None
                        if key not in fits:
                            fits[key] = model_type().fit(train)
                        sources[session_seed] = fits[key]
                local = [name for name in models if source_name == 'none' or name != 'random']
                print(f'{app} / {platform}: historical={source_name}, {len(local) * len(session_seeds)} sessions', flush=True)
                results = Parallel(n_jobs=jobs, return_as='generator_unordered')(
                    delayed(_session)(name, pool, steps, initial, session_seed, sources[session_seed],
                                      evaluation=True, random_state=session_seed)
                    for name in local for session_seed in session_seeds)
                for result in results:
                    tables = result.pop('_evaluation')
                    identity = dict(application=app, platform=platform, pretrained=source_name,
                                    unseen_values=None if source_name == 'none' else unseen,
                                    model=result['model'], seed=result['seed'],
                                    workload=pool['workload'].iloc[0] if 'workload' in pool else '')
                    save_tables(output / 'evaluation', tables, identity, append=True)
    print(f'Saved evaluation CSVs in {output}', flush=True)
