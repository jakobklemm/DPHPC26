#!/usr/bin/env python3
"""Build llama.cpp on the host with clang and benchmark a local GGUF model."""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from filter_llama_performance_flags import find_cuda_compiler, profile_prerequisite_error


ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(directory):
    revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    if revision.returncode:
        return None
    changes = subprocess.run(["git", "-C", str(directory), "status", "--porcelain"],
                             capture_output=True, text=True, check=True)
    return {"commit": revision.stdout.strip(), "dirty": bool(changes.stdout.strip())}


def run_logged(command, log, *, env=None, cwd=None):
    print("+", shlex.join(map(str, command)), flush=True)
    with log.open("a", encoding="utf-8") as destination:
        destination.write("+ " + shlex.join(map(str, command)) + "\n")
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                destination.write(line)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def benchmark_args(binary, model, args, repetitions):
    return [str(binary), "-m", str(model), "-ngl", str(args.gpu_layers),
            "-p", str(args.prompt), "-n", str(args.generate),
            "-r", str(repetitions), "-o", "json"]


def verify_benchmark(path, device):
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data or not all(
        isinstance(row, dict) and device.lower() in row.get("backends", "").lower()
        for row in data
    ):
        raise RuntimeError(f"Benchmark did not report {device} for every result; see {path}")


def verify_metrics(path, required=()):
    metrics = json.loads(path.read_text())
    if (not isinstance(metrics, dict) or not metrics
            or any(name not in metrics for name in required)):
        raise RuntimeError(f"Missing required metrics in {path}")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_assignments(parser, args, label):
    result = {}
    for item in args:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=[^\r\n]*", item):
            parser.error(f"{label} must be NAME=VALUE: {item!r}")
        name, value = item.split("=", 1)
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "llama.cpp")
    parser.add_argument("--model", type=Path, default=ROOT / "model.gguf")
    parser.add_argument("--flags-file", type=Path, default=ROOT / "flags.json")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "bench-results")
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--model-sha256", help="Precomputed model hash for repeated runs")
    parser.add_argument("--preflight-done", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--system-info", type=Path, help="Host snapshot prepared once by the runner")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--runtime-arg", action="append", default=[])
    parser.add_argument("--compiler-flag", action="append", default=[])
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--gpu-layers", type=int,
                        help="Defaults to 0 on CPU, 99 on GPU")
    args = parser.parse_args()
    if args.gpu_layers is None:
        args.gpu_layers = 0 if args.device == "cpu" else 99
    source = args.source.expanduser().resolve()
    model = args.model.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file() or not model.is_file():
        parser.error("--source must be a llama.cpp checkout and --model a GGUF file")
    cmake = parse_assignments(parser, args.cmake_arg, "--cmake-arg")
    runtime = parse_assignments(parser, args.runtime_arg, "--runtime-arg")
    if args.device == "cuda" and not cmake.get("CMAKE_CUDA_ARCHITECTURES"):
        parser.error("CUDA requires CMAKE_CUDA_ARCHITECTURES")
    if args.device != "cuda" and "CMAKE_CUDA_ARCHITECTURES" in cmake:
        parser.error("CUDA architecture requires --device cuda")
    if (cmake.get("GGML_CUDA", "OFF") == "ON") != (args.device == "cuda") or (
        cmake.get("GGML_VULKAN", "OFF") == "ON") != (args.device == "vulkan"):
        parser.error("GPU backend CMake flags must match --device")
    if args.device == "cpu" and args.gpu_layers != 0:
        parser.error("CPU runs require --gpu-layers 0")
    if any(not re.fullmatch(r"-[^\s]+", flag) for flag in args.compiler_flag):
        parser.error("--compiler-flag must be a single compiler switch")
    if any(flag in {"-O0", "-Ofast", "-ffast-math"} for flag in args.compiler_flag):
        parser.error("-O0 is unoptimized; fast-math/Ofast conflicts with ggml non-finite arithmetic")
    if any(not name.startswith("GGML_VK_") for name in runtime):
        parser.error("--runtime-arg only accepts GGML_VK_* settings")
    flags_file = args.flags_file.expanduser().resolve()
    if not flags_file.is_file():
        parser.error(f"Flag catalog not found: {flags_file}")
    with flags_file.open(encoding="utf-8") as source_flags:
        rules = json.load(source_flags).get("invalid_combinations", [])
    if any(all(cmake.get(name) == value for name, value in rule.items()) for rule in rules):
        parser.error("invalid CMake flag combination")
    if not shutil.which("clang") or not shutil.which("clang++"):
        parser.error("clang and clang++ are required")
    if args.device == "cuda" and not find_cuda_compiler():
        parser.error("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc; "
                     "install the CUDA development toolkit on this host")
    if not args.preflight_done:
        problem = profile_prerequisite_error(args.device)
        if problem:
            parser.error(problem)
    if args.model_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.model_sha256):
        parser.error("--model-sha256 must be a SHA-256 hex digest")

    started = datetime.now(timezone.utc)
    output = results / started.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    print(f"Run directory: {output}", flush=True)
    if args.system_info:
        (output / "system.json").write_bytes(args.system_info.read_bytes())
    else:
        (output / "system.json").write_text(json.dumps({
            "CPU Info": {"Architecture": platform.machine()},
            "GPU Backends": {args.device.capitalize(): {"version": "enabled"}}
            if args.device != "cpu" else {},
        }, indent=2) + "\n")
    (output / "source.yml").write_text("project_name: llama.cpp\n")
    record = {"run_timestamp_utc": started.isoformat(), "source": str(source),
              "source_git": git_state(source), "model": str(model),
              "model_sha256": args.model_sha256 or sha256_file(model), "device": args.device,
              "flags_file": str(flags_file), "flags_sha256": sha256_file(flags_file),
              "run_id": output.name,
              "cmake_definitions_in_order": ["-D" + arg for arg in args.cmake_arg],
              "cmake": cmake, "runtime": runtime, "compiler_flags": args.compiler_flag,
              "profiling": {"cpu_perf": True, "gpu_hpctoolkit": args.device == "cuda"},
              "stages": {"benchmark": "pending", "cpu_perf": "pending",
                         "gpu_hpctoolkit": "pending" if args.device == "cuda" else "not_applicable"},
              "script_sha256": sha256_file(Path(__file__).resolve())}
    record_file = output / "build.json"

    def save():
        record_file.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        optimization = [] if any(flag.startswith("-O") for flag in args.compiler_flag) else ["-O3"]
        flags = ["-g", "-DNDEBUG", *optimization, *args.compiler_flag]
        configure = ["cmake", "-S", str(source), "-B", str(build),
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang",
                     "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_C_FLAGS_RELEASE=" + " ".join(flags),
                     "-DCMAKE_CXX_FLAGS_RELEASE=" + " ".join(flags),
                     "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                     *[f"-D{name}={value}" for name, value in cmake.items()]]
        if args.device == "cuda":
            configure.extend(["-DCMAKE_CUDA_HOST_COMPILER=clang++",
                              "-DCMAKE_CUDA_FLAGS=-lineinfo"])
        record["configure_command"] = configure
        build_command = ["cmake", "--build", str(build), "--target", "llama-bench",
                         "-j", str(args.workers)]
        record["build_command"] = build_command
        save()
        start = time.monotonic()
        run_logged(configure, output / "build.log")
        run_logged(build_command, output / "build.log")
        record["build_time_seconds"] = round(time.monotonic() - start, 3)
        binary = build / "bin" / "llama-bench"
        if not binary.is_file():
            raise RuntimeError(f"Built benchmark not found at {binary}")
        env = os.environ.copy()
        env.update(runtime)
        bench = benchmark_args(binary, model, args, args.repetitions)
        record["benchmark_command"] = bench
        record["runtime_environment"] = runtime
        save()
        benchmark_file = output / "benchmark.json"
        with benchmark_file.open("w") as stdout, (output / "benchmark.log").open("w") as stderr:
            start = time.monotonic()
            subprocess.run(bench, env=env, stdout=stdout, stderr=stderr, check=True)
        verify_benchmark(benchmark_file, args.device)
        record["benchmark_wall_seconds"] = round(time.monotonic() - start, 3)
        record["stages"]["benchmark"] = "completed"
        save()
        profile(args, output, env, bench, record, save)
        print(f"CPU metrics: {output / 'cpu-metrics.json'}", flush=True)
        if args.device == "cuda":
            print(f"GPU metrics: {output / 'gpu-metrics.json'}", flush=True)
        record["status"] = "completed"
        save()
        print(f"Results: {benchmark_file}", flush=True)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        save()
        print(f"Failed: {error}\nLogs and generated files: {output}", file=sys.stderr, flush=True)
        return 1
    return 0


def profile(args, output, env, bench, record, save):
    cpu_dir = output / "profiles" / "cpu"
    cpu_dir.mkdir(parents=True)
    run_logged([sys.executable, str(ROOT / "collect_cpu_metrics.py"),
                "-o", str(output / "cpu-metrics.json"),
                "--benchmark-output", str(cpu_dir / "benchmark.json"),
                "--log", str(cpu_dir / "benchmark.log"), "--", *bench],
               output / "cpu-perf.log", env=env)
    verify_benchmark(cpu_dir / "benchmark.json", args.device)
    verify_metrics(output / "cpu-metrics.json", ("cycles", "instructions"))
    record["stages"]["cpu_perf"] = "completed"
    save()
    if args.device != "cuda":
        return
    gpu_dir = output / "profiles" / "gpu-operations"
    gpu_dir.mkdir(parents=True)
    measurements = gpu_dir / "measurements"
    database = gpu_dir / "database"
    with (gpu_dir / "benchmark.json").open("w") as stdout, (
        gpu_dir / "benchmark.log").open("w") as stderr:
        subprocess.run(["hpcrun", "-o", str(measurements), "-e", "gpu=cuda", *bench],
                       env=env, stdout=stdout, stderr=stderr, check=True)
    verify_benchmark(gpu_dir / "benchmark.json", "cuda")
    run_logged(["hpcstruct", "-j", str(args.workers), str(measurements)],
               gpu_dir / "analysis.log")
    run_logged(["hpcprof", "-o", str(database), str(measurements)], gpu_dir / "analysis.log")
    run_logged([sys.executable, str(ROOT / "extract_gpu_metrics.py"), str(database),
                "-o", str(output / "gpu-metrics.json")], gpu_dir / "analysis.log")
    verify_metrics(output / "gpu-metrics.json")
    record["stages"]["gpu_hpctoolkit"] = "completed"
    save()
    shutil.rmtree(measurements)
    shutil.rmtree(database)


if __name__ == "__main__":
    sys.exit(main())
