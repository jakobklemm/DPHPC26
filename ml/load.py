"""
Step 1: read the benchmark runs of all applications into a dataframe.

The benchmarking scripts write one folder per build. The folders can be given
as they are or still zipped. Zips are read in place, also when they contain
many runs (e.g. gromacs_xaas_<date>.zip: xaas/<machine>/<run>/...).
Only four small files of each folder are used:
    source.yml           project_name      -> which application
    system.json          XaaS system discovery of the machine -> hardware features
    build-<run id>.json  build configuration
    benchmark.json       benchmark output  -> performance

What is application specific (how to read the build configuration and the
performance, options that depend on the hardware) lives in one module per
application: load_llama.py, load_gromacs.py. To add an application, write such
a module and add it to APPS below.

The result is a table with one row per distinct build and platform:
    application | platform | cfg.<option>... | run_id | performance | hw.<feature>... | runs | relative
Runs of the same build on the same platform are merged into one row (see `load_table`),
their differences are pure noise, and we want each build to have the same weight
"""

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from . import load_gromacs, load_llama

HW = "hw."
CFG = "cfg."
DEFAULT = "default"

# project_name in source.yml -> module that knows how to read that application
APPS = {app.APPLICATION: app for app in [load_llama, load_gromacs]}

# Options whose value follows from the hardware, as {option: rule(hardware) -> value}, learning them would not work.
# A rule returning None means the option is not passed.
HARDWARE_DETERMINED = {
    option: (lambda hw, rule=rule: rule(hw) or DEFAULT)
    for app in APPS.values() for option, rule in app.HARDWARE_DETERMINED.items()
}


# Can a configuration be built and run on this hardware? Excludes e.g. GMX_SIMD=AVX_512 on an ARM CPU.
# Every application checks only its own options, so this works without knowing the application.
def valid(config, hardware):
    return all(app.valid(config, hardware) for app in APPS.values())


# files to analyze
RUN_FILES = ["source.yml", "system.json", "build.json", "benchmark.json"]


# --- Finding runs ----------------------------------------------------------------

# on the path get all the 4 files per run
# each run is a dict {"source.yml": text, "system.json": text, "build.json": text, "benchmark.json": text}
# is a generator! hands out one run at a time
def read_runs(paths):
    for path in map(Path, paths):
        if path.suffix == ".zip":
            yield from _runs_in_zip(path)
        else:
            # Extracted run folders and zipped ones can sit side by side.
            yield from _runs_in_folder(path)
            for zip_path in sorted(path.rglob("*.zip")):
                yield from _runs_in_zip(zip_path)


# we found a folder with a run and build the dict
def _runs_in_folder(folder):
    for system in sorted(folder.rglob("system.json")):
        run = system.parent
        builds = list(run.glob("build-*.json"))
        files = {
            "source.yml": run / "source.yml",
            "system.json": system,
            "build.json": builds[0] if builds else run / "build.json",
            "benchmark.json": run / "benchmark.json",
        }
        if all(f.is_file() for f in files.values()):
            yield {name: f.read_text() for name, f in files.items()}

# we found a zip with a run and build the dict
def _runs_in_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for system in sorted(n for n in names if n.endswith("system.json")):
            prefix = system[: -len("system.json")]
            # Files directly in the run folder, not in its subfolders.
            top_level = {n[len(prefix):]: n for n in names
                         if n.startswith(prefix) and "/" not in n[len(prefix):]}
            top_level["build.json"] = next(
                (n for name, n in top_level.items() if re.fullmatch(r"build-.*\.json", name)), None)
            if all(top_level.get(name) for name in RUN_FILES):
                yield {name: z.read(top_level[name]).decode() for name in RUN_FILES}


# --- Turning one run into one row ------------------------------------------------

# extract name of the application from source.yml
def application(source_yml):
    return re.search(r"^project_name:\s*(\S+)", source_yml, re.MULTILINE).group(1)


# GPUs by vendor, {"nvidia": [...]}; some machines report a list of messages instead
def _accelerators(system):
    accelerators = system.get("Accelerators", {}).get("Accelerators", {})
    return accelerators if isinstance(accelerators, dict) else {}

# extract hardware features from system.json
def hardware_features(system):
    cpu = system.get("CPU Info", {})
    vectorization = set(cpu.get("Supported Vectorizations", []))
    model_name = cpu.get("Model Name", "")
    nvidia_gpus = _accelerators(system).get("nvidia", [])
    backends = system.get("GPU Backends", {})

    def backend_available(name):
        return bool(backends.get(name, {}).get("version"))

    return {
        "cpu_arch": cpu.get("Architecture", "unknown"),
        "cpu_vendor": ("AMD" if "AMD" in model_name else "Intel" if "Intel" in model_name
                       else "Apple" if "Apple" in model_name else "other"),
        "avx2": "avx2" in vectorization,
        "avx512": any(v.startswith("avx512") for v in vectorization),
        "sve": "sve" in vectorization,
        "neon": "neon" in vectorization or "asimd" in vectorization,
        "n_gpus": len(nvidia_gpus),
        "gpu_vendor": "nvidia" if nvidia_gpus else "none",
        "gpu_compute_capability": float(nvidia_gpus[0]["compute_cap"]) if nvidia_gpus else 0.0,
        "cuda": backend_available("CUDA"),
        "hip": backend_available("HIP"),
        "sycl": backend_available("SYCL"),
    }

# extract human readable machine name, only used for printing
def platform_name(system):
    cpu = system.get("CPU Info", {}).get("Model Name", "unknown CPU")
    gpus = [g["gpu"] for g in _accelerators(system).get("nvidia", [])]
    return " + ".join([cpu] + gpus)


# compute one row of the dataframe
def parse_run(files):
    system = json.loads(files["system.json"])
    build = json.loads(files["build.json"])
    app = application(files["source.yml"])
    if app not in APPS:
        print(f"skipping run {build.get('run_id', '?')}: no loader for application {app!r}")
        return None
    row = {
        "run_id": build.get("run_id", ""),
        "application": app,
        "platform": platform_name(system),
        "performance": APPS[app].performance(json.loads(files["benchmark.json"])),
    }
    row.update({HW + k: v for k, v in hardware_features(system).items()})
    row.update({CFG + k: v for k, v in APPS[app].build_config(build).items()})
    return row


# --- The table -------------------------------------------------------------------

# reads all runs (optionally only of one application) into a dataframe
def load_table(paths, app=None):
    """
    The same build measured more than once on a platform becomes one row with
    the mean performance; `runs` says how many measurements it is based on.

    `relative` is the performance relative to the best run of the same
    application on the same platform: 1.0 is the best measured build there,
    0.9 is 10 % slower.

    """
    rows = [parse_run(files) for files in read_runs(paths)]
    rows = [r for r in rows if r and (app is None or r["application"] == app)]
    if not rows:
        raise ValueError(f"no runs{' of ' + app if app else ''} found in {list(map(str, paths))}")
    table = pd.DataFrame(rows)

    # The same run can be found twice, e.g. as a zip and as its extracted folder.
    table = table.drop_duplicates("run_id")

    # An option missing from a run was not passed, so CMake used its default.
    # (Options of other applications are "default" everywhere and dropped before training.)
    cfg_cols = [c for c in table.columns if c.startswith(CFG)]
    table[cfg_cols] = table[cfg_cols].fillna(DEFAULT)

    # Average repeated measurements of the same build on the same platform.
    keys = ["application", "platform"] + cfg_cols
    aggregate = {c: "first" for c in table.columns if c not in keys}
    aggregate.update(run_id=",".join, performance="mean")
    table = table.groupby(keys, as_index=False).agg(aggregate)
    table["runs"] = table["run_id"].str.count(",") + 1

    best = table.groupby(["application", "platform"])["performance"].transform("max")
    table["relative"] = table["performance"] / best
    return table


# only the options of one application: the others are "default" in all its rows
def app_options(table):
    return table.loc[:, (table != DEFAULT).any() | ~table.columns.str.startswith(CFG)]


def columns(table, prefix):
    return [c for c in table.columns if c.startswith(prefix)]


# cfg.GGML_LTO" -> "GGML_LTO"; names without a prefix stay as they are.
def strip(column):
    return column.split(".", 1)[-1]


# a configuration the way the application's build takes it, e.g. "-DGGML_LTO=ON"
def describe(app, config):
    return " ".join(APPS[app].build_argument(o, v) for o, v in config.items() if v != DEFAULT)


# Print the table for a look at the data. From the repository root: python -m ml.load path_to_a_folder_with_all_zips_or_folders
if __name__ == "__main__":
    import sys

    table = load_table(sys.argv[1:] or ["../download"])
    for app, runs in table.groupby("application"):
        runs = app_options(runs)
        print(f"== {app}: {len(runs)} rows (one per distinct build and platform), "
              f"{runs['platform'].nunique()} platform(s)\n")
        print(runs.head(3).T.to_string(max_colwidth=25))

        # The interesting part: only the options that differ between builds.
        varying = [c for c in columns(runs, CFG) if runs[c].nunique() > 1]
        print("\nOptions that differ between builds:\n")
        print(runs[["platform"] + varying + ["runs", "performance", "relative"]]
              .rename(columns=strip).sort_values(["platform", "relative"], ascending=[True, False])
              .round(3).to_string(index=False), end="\n\n")
