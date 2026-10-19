"""
Step 1: read the benchmark runs into a dataframe. currently works for llama

The benchmarking script (llama_gpu_bench.py) writes one folder per build. 
The folders can be given as they are or still zipped. Zips are read in place. 
Only four small files of each folder are used:
    source.yml           project_name      -> which application
    system.json          XaaS system discovery of the machine -> hardware features
    build-<run id>.json  cmake_definitions_in_order -> build configuration
    benchmark.json       benchmark output  -> performance

The result is a table with one row per distinct build and platform:
    application | platform | cfg.<option>... | run_id | performance | hw.<feature>... | runs | relative
Runs of the same build on the same platform are merged into one row (see `load_table`), 
their differerences are pure nose, and we want each build to have the sasme weight

"""

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

HW = "hw."
CFG = "cfg."
DEFAULT = "default"

# The default value CMake uses for an option when the build does not pass it.
# to detect that a build that does not pass GGML_LTO and a build that passes -DGGML_LTO=OFF are the same build
CMAKE_DEFAULTS = {
    "llama.cpp": {"GGML_LTO": "OFF", "GGML_OPENMP": "ON", "GGML_CUDA_FA": "ON"},
}

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

# extract hardware features from system.json
def hardware_features(system):
    cpu = system.get("CPU Info", {})
    vectorization = set(cpu.get("Supported Vectorizations", []))
    model_name = cpu.get("Model Name", "")
    nvidia_gpus = system.get("Accelerators", {}).get("Accelerators", {}).get("nvidia", [])
    backends = system.get("GPU Backends", {})

    def backend_available(name):
        return bool(backends.get(name, {}).get("version"))

    return {
        "cpu_arch": cpu.get("Architecture", "unknown"),
        "cpu_vendor": "AMD" if "AMD" in model_name else "Intel" if "Intel" in model_name else "other",
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
    gpus = [g["gpu"] for g in system.get("Accelerators", {}).get("Accelerators", {}).get("nvidia", [])]
    return " + ".join([cpu] + gpus)

# extract build onfig from build-<run>.json
def build_config(build):
    config = {}
    for flag in build["cmake_definitions_in_order"]:
        name, _, value = flag[len("-D"):].partition("=")
        config[name] = value.strip('"')
    return config

# extract performance metrics from benchmark.json
def llama_bench_performance(benchmark):
    """
    benchmark.json holds one entry per test, currently for llama two:
      - prompt processing (n_prompt=512): reads a prompt, compute-bound
      - token generation  (n_gen=1024):   writes an answer, memory-bound
    """
    # The tests are combined into the total time for the whole benchmark
    # workload (here: read 512 tokens, then generate 1024). Each test thus
    # counts as much as the time it takes: generation takes ~10.5 s and
    # prompt processing ~1.3 s, so generation weighs about 8 times more.
    total_seconds = sum(test["avg_ns"] for test in benchmark) / 1e9
    total_tokens = sum(test["n_prompt"] + test["n_gen"] for test in benchmark)
    # Returned as tokens per second over the whole workload, so that higher is
    # better like for the other applications.
    return total_tokens / total_seconds


# How to get one performance number (higher is better) out of each
# application's benchmark.json. Add the other applications here.
PERFORMANCE = {
    "llama.cpp": llama_bench_performance,
}

# compute one row of the dataframe
def parse_run(files):
    system = json.loads(files["system.json"])
    build = json.loads(files["build.json"])
    app = application(files["source.yml"])
    row = {
        "run_id": build.get("run_id", ""),
        "application": app,
        "platform": platform_name(system),
        "performance": PERFORMANCE[app](json.loads(files["benchmark.json"])),
    }
    row.update({HW + k: v for k, v in hardware_features(system).items()})
    config = {**CMAKE_DEFAULTS.get(app, {}), **build_config(build)}
    row.update({CFG + k: v for k, v in config.items()})
    return row


# --- The table -------------------------------------------------------------------

# reads all runs for one application into a datafrae
def load_table(paths, app=None):
    """
    The same build measured more than once on a platform becomes one row with
    the mean performance; `runs` says how many measurements it is based on.

    `relative` is the performance relative to the best run on the same
    platform: 1.0 is the best measured build there, 0.9 is 10 % slower.

    """
    rows = [parse_run(files) for files in read_runs(paths)]
    rows = [r for r in rows if app is None or r["application"] == app]
    if not rows:
        raise ValueError(f"no runs{' of ' + app if app else ''} found in {list(map(str, paths))}")
    table = pd.DataFrame(rows)

    # The same run can be found twice, e.g. as a zip and as its extracted folder.
    table = table.drop_duplicates("run_id")

    # An option missing from a run was not passed, so CMake used its default.
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


def columns(table, prefix):
    return [c for c in table.columns if c.startswith(prefix)]


# cfg.GGML_LTO" -> "GGML_LTO"; names without a prefix stay as they are.
def strip(column):
    return column.split(".", 1)[-1]





# Print the table for a look at the data. From the repository root: python -m ml.load path_to_a_folder_with_all_zips_or_folders
if __name__ == "__main__":
    import sys

    table = load_table(sys.argv[1:] or ["../download"])
    print(f"{len(table)} rows (one per distinct build and platform), {len(table.columns)} columns\n")

    print(table.T.to_string(max_colwidth=25))

    # The interesting part: only the options that differ between builds.
    varying = [c for c in columns(table, CFG) if table[c].nunique() > 1]
    print("\nOptions that differ between builds:\n")
    print(table[varying + ["runs", "performance", "relative"]]
          .sort_values("relative", ascending=False).round(3).to_string(index=False))
