"""Live active sampling of llama.cpp on this machine (ml.main active-sampling).

run_session repeats: LiveActiveSampler proposes a build, benchmark() builds and
measures it, the sampler refits. Each build gets a results folder that later
loads as training data.

Work folder layout (container setup creates it; DATA_DIR, default /data):
  llama.cpp/   pinned source checkout
  gguf/        inference weights (GGUF_NAME)
  build/       build directories, one per session
"""

import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .active_sampling import LiveActiveSampler
from .load import CFG, DEFAULT, app_options, columns, describe, hardware_features, platform_name, strip
from .load_llama import APPLICATION, performance
from .validation import compiler_flags


GGUF_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sampling_space(values, device):
    """Keep learned tuning choices, but fix the backend for this session."""
    values = {name: list(choices) for name, choices in values.items()}
    if device == "cpu":
        values = {name: choices for name, choices in values.items()
                  if not name.startswith(("CMAKE_CUDA_", "GGML_CUDA_"))}
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


def benchmark(config, data, build, output, system, device="cuda"):
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
        write_json(output / "build.json", record)


def detect_device(hardware):
    """CUDA needs the toolkit ("cuda") and a visible NVIDIA GPU; the CUDA image always has the toolkit."""
    gpus = hardware.get("n_gpus")
    has_gpu = hardware.get("gpu_vendor") == "nvidia" and isinstance(gpus, int) and gpus > 0
    return "cuda" if hardware.get("cuda") is True and has_gpu else "cpu"


def run_session(model_type, system, table, *, pretrain, steps, initial, seed, device=None,
                workdir, output=None, values=None, settings=None):
    """Benchmark `steps` builds proposed by model_type (None: random); return the results folder.

    table: historical runs for the flag space; pretrain(platform): model fitted on
    the other machines, or None. values replaces the flag space from table.
    """
    work = Path(workdir).expanduser().resolve()
    if not (work / "llama.cpp" / "CMakeLists.txt").is_file() or not (work / "gguf" / GGUF_NAME).is_file():
        raise ValueError(f"llama.cpp source or gguf/{GGUF_NAME} missing in {work}; "
                         "run container setup or pass --workdir")
    platform, hardware = platform_name(system), hardware_features(system)
    device = device or detect_device(hardware)

    # Flag space from the benchmarks (or values), backend fixed to the device
    if values is None:
        runs = app_options(table)
        values = {strip(c): sorted(runs[c].unique()) for c in columns(runs, CFG)}
    values = sampling_space(values, device)
    pretrained = pretrain(platform)
    sampler = LiveActiveSampler(model_type, values, hardware, APPLICATION,
                                initial=initial, seed=seed, pretrained=pretrained)

    # One results folder per session
    session = "active-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output or work / "llama" / "results").expanduser().resolve() / session
    output.mkdir(parents=True)
    build = work / "build" / session
    write_json(output / "session.json", {**(settings or {}), "platform": platform,
                                         "device": device, "space": values})
    settings = settings or {}
    model = settings.get("model") or ("random" if model_type is None else model_type.__name__)
    source = settings.get("pretrained") or (type(pretrained).__name__ if pretrained else "none")
    print(f"== {APPLICATION} on {platform} ({device}): {steps} steps, model {model}, "
          f"pretrained {source}\n   results: {output}", flush=True)

    best_score = 0.0
    for step in range(1, steps + 1):
        config = sampler.ask()
        print(f"\n-- build {step}/{steps} ({sampler.source}): {describe(APPLICATION, config)}", flush=True)
        step_output = output / f"{step:04d}"
        step_output.mkdir()
        try:
            score = benchmark(config, work, build, step_output, system, device)
        except RuntimeError as error:
            # Failed builds count as a step and are not proposed again
            print(f"   failed: {error}", flush=True)
            sampler.fail(config)
            continue
        sampler.tell(config, score)
        best_score = max(best_score, score)
        write_json(output / "best.json", {"steps": step, "config": sampler.recommend(),
                                          "performance": best_score})
        print(f"   {score:.3f} tokens/s (best so far {best_score:.3f})", flush=True)

    if not sampler.measured:
        raise ValueError(f"no build succeeded; see {output}")
    print(f"\nbest measured ({best_score:.3f} tokens/s): {describe(APPLICATION, sampler.recommend())}")
    if sampler.model is not None:
        config, score = sampler.predict()
        print(f"model pick (predicted {score:.3f} of best measured, unmeasured if different): "
              f"{describe(APPLICATION, config)}")
    print(f"details: {output / 'best.json'}")
    return output
