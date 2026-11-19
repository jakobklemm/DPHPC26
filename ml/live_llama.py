"""Live active sampling of llama.cpp on this machine (ml.main active-sampling).

The session itself is in live.py; this module builds and benchmarks one
configuration like llama_gpu_bench-profile.py.

Work folder layout (container setup creates it; DATA_DIR, default /data):
  llama.cpp/   pinned source checkout
  gguf/        inference weights (GGUF_NAME)
  build/       build directories, one per session
"""

import json
import math
import shlex
import subprocess
import time

from .load import DEFAULT
from .load_llama import APPLICATION, performance
from .live import write_json
from .validation import compiler_flags


GGUF_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
UNIT = "tokens/s"
RESULTS = "llama"   # results folder: <workdir>/llama/results
WORKLOAD = json.dumps([[0, 128], [512, 0]])  # pp512/tg128, as the loader writes it


def prepare(work, device):
    """The work folder holds the source and the weights; the benchmark context is the folder itself."""
    if not (work / "llama.cpp" / "CMakeLists.txt").is_file() or not (work / "gguf" / GGUF_NAME).is_file():
        raise ValueError(f"llama.cpp source or gguf/{GGUF_NAME} missing in {work}; "
                         "run container setup or pass --workdir")
    return work


def workload(context):
    return WORKLOAD


def sampling_space(values, device, hardware=None):
    """Keep learned tuning choices, but fix the backend for this session."""
    values = {name: list(choices) for name, choices in values.items()}
    if device == "cpu":
        values = {name: choices for name, choices in values.items()
                  if not name.startswith(("CMAKE_CUDA_", "GGML_CUDA_", "CMAKE_VULKAN_",
                                          "GGML_VULKAN_", "GGML_VK_"))}
    values.update(GGML_CPU=["ON"], GGML_CUDA=["ON" if device == "cuda" else "OFF"],
                  GGML_VULKAN=["OFF"], GGML_HIP=["OFF"], GGML_SYCL=["OFF"], GGML_METAL=["OFF"])
    return values


def commands(config, data, build, device="cuda"):
    compiler = compiler_flags(APPLICATION)
    cmake = {name: str(value) for name, value in config.items()
             if name not in compiler and value != DEFAULT}
    flags = [str(value) for name, value in config.items()
             if name in compiler and value != DEFAULT]
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
    if device == "cuda":
        configure.extend(["-DCMAKE_CUDA_HOST_COMPILER=clang++", "-DCMAKE_CUDA_FLAGS=-lineinfo"])
    compile_command = ["cmake", "--build", str(build), "--target", "llama-bench", "-j", "4"]
    benchmark = [str(build / "bin" / "llama-bench"), "-m", str(data / "gguf" / GGUF_NAME),
                 "-ngl", "99" if device == "cuda" else "0",
                 "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
    return configure, compile_command, benchmark, flags


def benchmark(config, data, build, output, system, device="cuda", *, timings=None):
    timings = timings if timings is not None else {}
    timings.update(configure_seconds=None, compile_seconds=None, benchmark_execution_seconds=None)

    def timed_run(key, command, **kwargs):
        start = time.perf_counter()
        try:
            return subprocess.run(command, **kwargs)
        finally:
            timings[key] = time.perf_counter() - start

    configure, compile_command, bench, flags = commands(config, data, build, device)
    record = {"run_id": f"{output.parent.name}-{output.name}", "config": config,
              "device": device, "source": str(data / "llama.cpp"), "model": str(data / "gguf" / GGUF_NAME),
              "cmake_definitions_in_order": [f"-D{k}={v}" for k, v in config.items()
                                             if k not in compiler_flags(APPLICATION) and v != DEFAULT],
              "compiler_flags": flags, "configure_command": configure,
              "build_command": compile_command, "benchmark_command": bench, "status": "running"}
    write_json(output / "system.json", system)
    (output / "source.yml").write_text("project_name: llama.cpp\n")
    write_json(output / "build.json", record)
    try:
        with (output / "build.log").open("w") as log:
            for key, command in (("configure_seconds", configure), ("compile_seconds", compile_command)):
                print("+", shlex.join(command), flush=True)
                timed_run(key, command, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Only publish benchmark.json after validation; the loader must not ingest failed runs.
        raw = output / "benchmark.raw.json"
        with raw.open("w") as stdout, (output / "benchmark.log").open("w") as stderr:
            timed_run('benchmark_execution_seconds', bench, stdout=stdout, stderr=stderr, check=True)
        results = json.loads(raw.read_text())
        expected = {(512, 0), (0, 128)}
        if (not isinstance(results, list) or len(results) != 2
                or {(row["n_prompt"], row["n_gen"]) for row in results} != expected
                or any(not math.isfinite(row["avg_ns"]) or row["avg_ns"] <= 0 for row in results)
                or any(device not in row.get("backends", "").lower() for row in results)
                or (device == "cpu" and any(row.get("n_gpu_layers") != 0 for row in results))):
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
        record.update(timings)
        write_json(output / "build.json", record)
