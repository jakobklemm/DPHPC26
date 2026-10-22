"""
Command line interface. Run from the repository root.

  train     read runs, train one model per application, save to models/<application>.joblib
  predict   build configuration for a machine, given its system.json
  suggest   configurations worth benchmarking next on a machine

--data takes run folders, folders containing run folders, and/or the run .zip files.

Examples:
  python -m ml.main train    --data runs/
  python -m ml.main predict  --app llama.cpp --system system.json
  python -m ml.main suggest  --app llama.cpp --system system.json --data runs/
  
"""

import argparse
import json
from pathlib import Path

# from .evaluate import leave_one_build_out, leave_one_platform_out
from .load import CFG, app_options, columns, describe, hardware_features, load_table, platform_name, strip
from .model import PerformanceModel


def model_path(models, app):
    return Path(models) / f"{app}.joblib"


# the configs in `table` as plain dicts {option: value}
def configs(table):
    cols = columns(table, CFG)
    return [{strip(c): row[c] for c in cols} for _, row in table[cols].iterrows()]


# Hardware features of the machine described by a system.json
def read_hardware(path):
    system = json.loads(Path(path).read_text())
    return platform_name(system), hardware_features(system)


def train(args):
    table = load_table(args.data) # load all available runs
    Path(args.models).mkdir(parents=True, exist_ok=True) # create a models folder
    
    for app, runs in table.groupby("application"): # for each application (e.g. llama)
        # removes build options that don't belong to the application being trained
        runs = app_options(runs)
        
        # train the model
        model = PerformanceModel().fit(runs)
        model.save(model_path(args.models, app))

        # print summary
        print(f"== {app}: {runs['runs'].sum()} runs of {len(runs)} distinct builds "
              f"on {runs['platform'].nunique()} platform(s) "
              f"-> {model_path(args.models, app)}")
        # Only the options that differ between runs, the rest is the same everywhere.
        varying = [c for c in columns(runs, CFG) if runs[c].nunique() > 1]
        print(runs[["platform"] + varying + ["runs", "performance", "relative"]]
              .rename(columns=strip).sort_values("relative", ascending=False)
              .round(3).to_string(index=False))
        print("\neffects learned by the model (vs. the reference value of each option):")
        print(model.effects().head(10).map("{:+.1%}".format).to_string(), end="\n\n")


def predict(args):
    model = PerformanceModel.load(model_path(args.models, args.app))
    _, hardware = read_hardware(args.system)
    config, predicted = model.predict(hardware)
    # Printed as build arguments (e.g. -DGGML_LTO=ON); options left at "default" are not passed.
    for argument in describe(args.app, config).split(" "):
        print(argument)
    print(f"# predicted performance: {predicted:.3f} of the best build on this machine")


def suggest(args):
    table = app_options(load_table(args.data, args.app))
    model = PerformanceModel().fit(table)
    platform, hardware = read_hardware(args.system)
    measured = configs(table[table["platform"] == platform])
    suggestions = model.suggest(hardware, measured, n=args.n)
    if not suggestions:
        print("every combination of the option values seen so far is already measured on this machine")
    for config, mean, spread in suggestions:
        print(f"predicted {mean:.3f} ± {spread:.3f}:  {describe(args.app, config)}")


# def evaluate(args):
#     table = load_table(args.data, args.app)
#
#     print("== leave one build out: predict each build without having seen it")
#     result = leave_one_build_out(table)
#     print(result.round(3).to_string(index=False))
#     print("\naverage:", result[["error", "guess"]].mean().round(3).to_dict(), end="\n\n")
#
#     print("== leave one platform out: predict each machine without having seen it")
#     if table["platform"].nunique() < 2:
#         print("needs runs from at least 2 platforms; found only:", *table["platform"].unique())
#         return
#     result = leave_one_platform_out(table)
#     print(result.round(3).to_string(index=False))
#     print("\naverage:", result[["error", "model", "baseline"]].mean().round(3).to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="models", help="folder with one saved model per application")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--data", nargs="+", required=True)
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--app", required=True)
    p.add_argument("--system", required=True, help="system.json of the target machine")
    p.set_defaults(func=predict)

    p = sub.add_parser("suggest")
    p.add_argument("--app", required=True)
    p.add_argument("--system", required=True, help="system.json of the target machine")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("-n", type=int, default=5, help="number of configurations to suggest")
    p.set_defaults(func=suggest)

    # p = sub.add_parser("evaluate")
    # p.add_argument("--app", required=True)
    # p.add_argument("--data", nargs="+", required=True)
    # p.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
