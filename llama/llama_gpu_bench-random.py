#!/usr/bin/env python3
"""Run a deterministic, resumable sweep of host-supported llama build settings."""

import argparse
import itertools
import json
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SEED = 20261021
PAIR_COUNT = 50
COMPILER = "g++"
CUDA_VERSION = "12.8"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUDA_ARCHITECTURES = "75"
WORKERS = 12
GENERATE = 1024
REPETITIONS = 5


def valid(cmake, rules):
    return not any(
        all(cmake.get(name) == value for name, value in rule.items())
        for rule in rules
    )


def execution_plan(options):
    """Generate single-option runs followed by 50 distinct two-option runs."""
    rng = random.Random(SEED)
    fixed = {name: values[0] for name, values in options["cmake"].items() if len(values) == 1}
    fixed["CMAKE_CUDA_ARCHITECTURES"] = CUDA_ARCHITECTURES
    ggml = [name for name, values in options["cmake"].items()
            if len(values) > 1 and "ON" in values]
    compiler_options = options["compiler"]
    rules = options.get("invalid_combinations", [])
    cmake_candidates = [("cmake", name) for name in ggml]
    compiler_candidates = [("compiler", name) for name in compiler_options]
    candidates = cmake_candidates + compiler_candidates

    def configuration(choices):
        cmake = fixed | {name: "ON" for kind, name in choices if kind == "cmake"}
        compiler_flags = {
            name: rng.choice(compiler_options[name])
            for kind, name in choices if kind == "compiler"
        }
        return {"compiler": COMPILER, "cuda_version": CUDA_VERSION,
                "cmake": cmake, "compiler_flags": compiler_flags}

    rng.shuffle(cmake_candidates)
    rng.shuffle(compiler_candidates)
    plan = []
    for candidate in cmake_candidates + compiler_candidates:
        cmake = fixed | ({candidate[1]: "ON"} if candidate[0] == "cmake" else {})
        if not valid(cmake, rules):
            raise ValueError(f"Cannot enable {candidate[1]} without an invalid combination")
        plan.append({"phase": "single", "config": configuration((candidate,))})

    pairs = [
        pair for pair in itertools.combinations(candidates, 2)
        if valid(fixed | {name: "ON" for kind, name in pair if kind == "cmake"}, rules)
    ]
    if len(pairs) < PAIR_COUNT:
        raise ValueError(f"Need {PAIR_COUNT} valid pairs, only {len(pairs)} available")
    for pair in rng.sample(pairs, PAIR_COUNT):
        plan.append({"phase": "pair", "config": configuration(pair)})
    return plan


def benchmark_command(args, root, results_dir, config):
    command = [
        sys.executable, str(ROOT / "llama_gpu_bench-profile.py"),
        "--xaas-root", str(root), "--model", str(args.model.expanduser()),
        "--source-config", str(args.source_config), "--deploy-config", str(args.deploy_config),
        "--results-dir", str(results_dir),
        "--cuda-version", CUDA_VERSION, "--cuda-image", CUDA_IMAGE,
        "--workers", str(WORKERS), "--generate", str(GENERATE),
        "--repetitions", str(REPETITIONS),
    ]
    for name, value in config["cmake"].items():
        command.extend(["--cmake-arg", f"{name}={value}"])
    for value in config["compiler_flags"].values():
        command.append(f"--compiler-flag={value}")
    return command


def load_records(path, plan):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or len(records) > len(plan):
        raise ValueError(f"Invalid execution list: {path}")
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or record.get("config") != plan[index]["config"]
                or not isinstance(record.get("results_directory"), str)
                or not record["results_directory"]
                or not isinstance(record.get("elapsed_seconds"), (int, float))):
            raise ValueError(f"Execution {index + 1} in {path} differs from the generated plan")
    return records


def save_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash during a write must leave the previous completed runs intact.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    ) as destination:
        temporary = Path(destination.name)
        json.dump(records, destination, indent=2)
        destination.write("\n")
    temporary.replace(path)


def run_benchmark(command, results_dir):
    benchmark_file = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("Results: "):
                benchmark_file = Path(line.removeprefix("Results: ").strip())
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    if (benchmark_file is None or benchmark_file.name != "benchmark.json"
            or not benchmark_file.is_file()
            or not benchmark_file.parent.is_relative_to(results_dir)):
        raise RuntimeError(
            "Benchmark completed without a result in the requested results directory"
        )
    return str(benchmark_file.parent.relative_to(results_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xaas-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-config", type=Path, default=Path("llama-source.yml"))
    parser.add_argument("--deploy-config", type=Path, default=Path("llama-deploy.yml"))
    parser.add_argument(
        "--results-dir", type=Path, help="Defaults to bench-results beside this script"
    )
    parser.add_argument(
        "--record-file", type=Path, help="Defaults to executions.json in results-dir"
    )
    parser.add_argument(
        "--flags-output", type=Path, help="Optionally save the host-filtered flag catalog"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the entire plan without building"
    )
    args = parser.parse_args()

    root = args.xaas_root.expanduser().resolve(strict=True)
    deploy = (root / args.deploy_config.expanduser()).resolve(strict=True)
    with deploy.open(encoding="utf-8") as source:
        configured_compiler = yaml.safe_load(source)["mode"]["predefined_config"]["compiler"]
    if configured_compiler != COMPILER:
        parser.error(f"Deploy compiler must be {COMPILER}, got {configured_compiler}")
    results_dir = (args.results_dir.expanduser().resolve() if args.results_dir is not None
                   else ROOT / "bench-results")
    record_file = (args.record_file.expanduser().resolve() if args.record_file is not None
                   else results_dir / "executions.json")

    with tempfile.TemporaryDirectory() as temporary:
        flags_output = (args.flags_output.expanduser().resolve() if args.flags_output is not None
                        else Path(temporary) / "flags.json")
        subprocess.run([
            sys.executable, str(ROOT / "filter_llama_performance_flags.py"),
            "--compiler", COMPILER, "--output", str(flags_output),
        ], check=True)
        with flags_output.open(encoding="utf-8") as source:
            plan = execution_plan(json.load(source))

    records = load_records(record_file, plan)
    print(f"Plan: {len(plan)} executions (seed {SEED}); {len(records)} completed", flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for index in range(len(records), len(plan)):
        config = plan[index]["config"]
        command = benchmark_command(args, root, results_dir, config)
        print(f"\nExecution {index + 1}/{len(plan)} ({plan[index]['phase']}): "
              f"{json.dumps(config)}", flush=True)
        print(shlex.join(command), flush=True)
        start = time.monotonic()
        directory = run_benchmark(command, results_dir)
        records.append({"config": config, "results_directory": directory,
                        "elapsed_seconds": round(time.monotonic() - start, 3)})
        save_records(record_file, records)
        print(f"Recorded execution {index + 1} in {record_file}", flush=True)


if __name__ == "__main__":
    main()
