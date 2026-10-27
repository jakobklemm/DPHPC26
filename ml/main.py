"""
Command line interface. Run from the repository root.

  train     read runs, train one model per application, save to models/<application>.joblib
  predict   build configuration for a machine, given its system.json
  suggest   configurations worth benchmarking next on a machine

Data defaults to each application's local results folder (llama/results or
gromacs/results). --data overrides it with local folders or ZIP files.
--app optionally selects one application.

Examples:
  python -m ml.main train
  python -m ml.main predict --system system.json
  python -m ml.main suggest --system system.json
  
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import leave_one_platform_out #, leave_one_build_out
from .load import APPS, HW, app_options, columns, describe, hardware_features, load_table, platform_name, strip
from .models.baseline_models import NoHardwareModel, RandomBaseline
from .models.ridge_regression import PerformanceModel
from .models.common import BaseModel, configs


MODEL_TYPES = {"hardware": PerformanceModel, "no-hardware": NoHardwareModel, "random": RandomBaseline}


def model_path(models, app, kind="hardware"):
    suffix = "" if kind == "hardware" else f".{kind}"
    return Path(models) / f"{app}{suffix}.joblib"


# Hardware features of the machine described by a system.json
def read_hardware(path):
    system = json.loads(Path(path).read_text())
    return platform_name(system), hardware_features(system)


def train(args):
    table = load_table(args.data, args.app)
    Path(args.models).mkdir(parents=True, exist_ok=True) # create a models folder
    
    for app, runs in table.groupby("application"): # for each application (e.g. llama)
        # removes build options that don't belong to the application being trained
        runs = app_options(runs)
        
        # train the model
        model = MODEL_TYPES[args.model]().fit(runs)
        model.save(model_path(args.models, app, args.model))

        # print summary
        print(f"== {app}: {runs['runs'].sum()} runs of {len(runs)} distinct builds "
              f"on {runs['platform'].nunique()} platform(s) "
              f"-> {model_path(args.models, app, args.model)}")
        if isinstance(model, RandomBaseline):
            print("random baseline: recorded option values; no performance model fitted\n")
            continue
        
        # Only the options that differ between runs, the rest is the same everywhere.
        # varying = [c for c in columns(runs, CFG) if runs[c].nunique() > 1]
        # print(runs[["platform"] + varying + ["runs", "performance", "relative"]].rename(columns=strip).sort_values("relative", ascending=False).round(3).to_string(index=False))
        
        # what the model learned
        print(f"model fit on the training data: explains {model.r2:.0%} of the measured performance differences, "
              f"estimated noise ±{model.noise:.1%}")
        if isinstance(model, NoHardwareModel):
            # The option-only fit is identical for every machine. Avoid the
            # full candidate-space enumeration used by the decision report.
            print()
            continue
        print("\nper machine and option: the value the model picks, then what each other value costs")
        for platform, machine in runs.groupby("platform"):
            hardware = {strip(c): machine[c].iloc[0] for c in columns(machine, HW)}
            decisions = model.decisions(hardware)
            print(f"\n  {platform}  ({len(machine)} builds measured)")
            if decisions.empty:
                print("    no alternative option values available")
                continue
            # options with the most expensive wrong choice first
            order = decisions.groupby("option")["cost"].min().sort_values().index
            for option in order:
                rows = decisions[decisions["option"] == option].sort_values("cost", ascending=False)
                print(f"    {option:20s} {rows['value'].iloc[0]}")
                for _, d in rows.iterrows():
                    also = "".join(f"  (with {o}={v})" for o, v in d["also"].items())
                    print(f"      {'':18s} {d['alternative']:16s} {d['cost']:+5.0%} ±{d['uncertainty']:.0%}{also}")
        print()


def predict(args):
    _, hardware = read_hardware(args.system)
    # One selected application or all saved models
    apps = [args.app] if args.app else [app for app in APPS if model_path(args.models, app, args.model).is_file()]
    if not apps:
        raise ValueError("no trained models found; run python -m ml.main train first")
    for app in apps:
        model = BaseModel.load(model_path(args.models, app, args.model))
        if isinstance(model, RandomBaseline):
            # A fresh generator avoids replaying the state saved during training.
            model.rng = np.random.default_rng(args.seed)
        config, predicted = model.predict(hardware)
        print(f"== {app}")
        for argument in describe(app, config).split(" "):
            print(argument)
        if predicted is not None:
            print(f"# predicted performance: {predicted:.3f} of the best build on this machine")


def suggest(args):
    table = load_table(args.data, args.app)
    platform, hardware = read_hardware(args.system)
    # Fresh model and measured builds per application
    for app, runs in table.groupby("application"):
        runs = app_options(runs)
        model = PerformanceModel().fit(runs)
        measured = configs(runs[runs["platform"] == platform])
        suggestions = model.suggest(hardware, measured, n=args.n)
        print(f"== {app}")
        if not suggestions:
            print("every combination of the option values seen so far is already measured on this machine")
        for config, mean, spread in suggestions:
            print(f"predicted {mean:.3f} ± {spread:.3f}:  {describe(app, config)}")


def evaluate(args):
    table = load_table(args.data, args.app)
    # print("== leave one build out: predict each build without having seen it")
    # result = leave_one_build_out(table)
    # print(result.round(3).to_string(index=False))
    # print("\naverage:", result[["error", "guess"]].mean().round(3).to_dict(), end="\n\n")

    # Separate validation per application
    for app, runs in table.groupby("application"):
        runs = app_options(runs)
        print(f"== {app}: leave one platform out")
        if runs["platform"].nunique() < 2:
            print("needs runs from at least 2 platforms; found only:", *runs["platform"].unique())
            continue
        result = leave_one_platform_out(runs, random_trials=args.random_trials, seed=args.seed)
        print(result.round(3).to_string(index=False))
        print("\naverage:", result[["error", "model", "baseline", "no_hardware", "random"]].mean().round(3).to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="models", help="folder with one saved model per application")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--model", choices=MODEL_TYPES, default="hardware", help="model to train; baseline files are saved separately")
    p.add_argument("--data", nargs="+", help="local folders/ZIPs; defaults to application results folders")
    p.add_argument("--app", choices=sorted(APPS), help="train only this application; default: all discovered applications")
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--model", choices=MODEL_TYPES, default="hardware", help="saved model to use")
    p.add_argument("--seed", type=int, help="reproducible random baseline selection")
    p.add_argument("--app", choices=sorted(APPS), help="default: all available trained applications")
    p.add_argument("--system", required=True, help="system.json of the target machine")
    p.set_defaults(func=predict)

    p = sub.add_parser("suggest")
    p.add_argument("--app", choices=sorted(APPS), help="default: all discovered applications")
    p.add_argument("--system", required=True, help="system.json of the target machine")
    p.add_argument("--data", nargs="+", help="local folders/ZIPs; defaults to application results folders")
    p.add_argument("-n", type=int, default=5, help="number of configurations to suggest")
    p.set_defaults(func=suggest)

    p = sub.add_parser("evaluate")
    p.add_argument("--random-trials", type=int, default=1000, help="random selections per held-out platform (default: 1000)")
    p.add_argument("--seed", type=int, default=0, help="random evaluation seed (default: 0)")
    p.add_argument("--app", choices=sorted(APPS), help="default: all discovered applications")
    p.add_argument("--data", nargs="+", help="local folders/ZIPs; defaults to application results folders")
    p.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
