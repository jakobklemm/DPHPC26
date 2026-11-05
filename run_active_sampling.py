#!/usr/bin/env python3
"""Run CUDA active sampling using the model and paths prepared by container setup."""

import argparse
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ml.active_sampling import LiveActiveSampler
from ml.load import DEFAULT, hardware_features
from ml.load_llama import COMPILER_FLAGS, performance
from ml.models import BaseModel, MODEL_TYPES


GGUF_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def commands(config, data, build):
    cmake = {name: str(value) for name, value in config.items()
             if name not in COMPILER_FLAGS and value != DEFAULT}
    flags = [str(value) for name, value in config.items()
             if name in COMPILER_FLAGS and value != DEFAULT]
    # Match llama_gpu_bench-profile.py, called by the training-data subset runner.
    optimization = [] if any(flag.startswith("-O") for flag in flags) else ["-O3"]
    release_flags = " ".join(["-g", "-DNDEBUG", *optimization, *flags])
    configure = ["cmake", "--fresh", "-S", str(data / "llama.cpp"), "-B", str(build),
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang",
                 "-DCMAKE_CXX_COMPILER=clang++", f"-DCMAKE_C_FLAGS_RELEASE={release_flags}",
                 f"-DCMAKE_CXX_FLAGS_RELEASE={release_flags}",
                 "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                 "-DLLAMA_BUILD_SERVER=OFF"]
    configure.extend(f"-D{name}={value}" for name, value in cmake.items())
    configure.extend(["-DCMAKE_CUDA_HOST_COMPILER=clang++", "-DCMAKE_CUDA_FLAGS=-lineinfo"])
    compile_command = ["cmake", "--build", str(build), "--target", "llama-bench", "-j", "4"]
    benchmark = [str(build / "bin" / "llama-bench"), "-m", str(data / "gguf" / GGUF_NAME),
                 "-ngl", "99", "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
    return configure, compile_command, benchmark, flags


def benchmark(config, data, build, output, system):
    configure, compile_command, bench, flags = commands(config, data, build)
    record = {"run_id": f"{output.parent.name}-{output.name}", "config": config,
              "device": "cuda", "source": str(data / "llama.cpp"), "model": str(data / "gguf" / GGUF_NAME),
              "cmake_definitions_in_order": [f"-D{k}={v}" for k, v in config.items()
                                             if k not in COMPILER_FLAGS and v != DEFAULT],
              "compiler_flags": flags, "configure_command": configure,
              "build_command": compile_command, "benchmark_command": bench, "status": "running"}
    write_json(output / "system.json", system)
    (output / "source.yml").write_text("project_name: llama.cpp\n")
    write_json(output / "build.json", record)
    try:
        with (output / "build.log").open("w") as log:
            for command in (configure, compile_command):
                print("+", shlex.join(command), flush=True)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Only publish benchmark.json after validation; the loader must not ingest failed runs.
        raw = output / "benchmark.raw.json"
        with raw.open("w") as stdout, (output / "benchmark.log").open("w") as stderr:
            subprocess.run(bench, stdout=stdout, stderr=stderr, check=True)
        results = json.loads(raw.read_text())
        expected = {(512, 0), (0, 128)}
        if (not isinstance(results, list) or len(results) != 2
                or {(row["n_prompt"], row["n_gen"]) for row in results} != expected
                or any(not math.isfinite(row["avg_ns"]) or row["avg_ns"] <= 0 for row in results)
                or any("cuda" not in row.get("backends", "").lower() for row in results)):
            raise ValueError("benchmark returned an unexpected workload, backend, or timing")
        score = performance(results)
        if not math.isfinite(score) or score <= 0:
            raise ValueError("benchmark performance must be positive and finite")
        raw.replace(output / "benchmark.json")
        record.update(status="completed", performance=score)
        return score
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        record.update(status="failed", error=str(error))
        raise RuntimeError(f"build/benchmark failed; see {output}: {error}") from error
    finally:
        write_json(output / "build.json", record)


def run(args):
    data = Path(os.environ.get("DATA_DIR", "/data")).expanduser().resolve()
    if not (data / "llama.cpp/CMakeLists.txt").is_file() or not (data / "gguf" / GGUF_NAME).is_file():
        raise ValueError("llama.cpp source or GGUF missing; run container setup first")
    system = json.loads((data / "system.json").read_text())
    suffix = "" if args.model == "hardware" else f".{args.model}"
    model_file = data / "models" / f"llama.cpp{suffix}.joblib"
    if not model_file.is_file():
        raise ValueError(f"missing {model_file}; run setup --model {args.model} first")
    saved = BaseModel.load(model_file)
    # Reuse setup's observed flag values; all validity/sampling decisions belong to ML.
    pretrained = None if args.model == "random" else saved
    sampler = LiveActiveSampler(None if args.model == "random" else MODEL_TYPES[args.model],
                                saved.values, hardware_features(system), "llama.cpp", pretrained=pretrained)
    session = "active-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = data / "llama/results" / session
    output.mkdir(parents=True)
    build = data / "build" / session
    write_json(output / "session.json", {**vars(args), "setup_model": str(model_file),
                                         "pretrained": pretrained is not None, "space": saved.values})
    print(f"Session: {output}\nSetup model: {model_file}", flush=True)
    best_score = 0
    for step in range(1, args.steps + 1):
        config = sampler.ask()
        step_output = output / f"{step:04d}"
        step_output.mkdir()
        score = benchmark(config, data, build, step_output, system)
        sampler.tell(config, score)
        best = sampler.recommend()
        best_score = max(best_score, score)
        write_json(output / "best.json", {"steps": step, "config": best,
                                          "performance": best_score})
        print(f"Step {step}/{args.steps}: {score:.3f} tokens/s", flush=True)
    print(f"Best measured build: {output / 'best.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, required=True, help="number of live builds to measure")
    parser.add_argument("--model", choices=MODEL_TYPES, default="hardware", help="same model kind used during setup")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"active-sampling: {error}\n")


if __name__ == "__main__":
    main()
