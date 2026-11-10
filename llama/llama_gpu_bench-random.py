#!/usr/bin/env python3
"""Run reproducible one-hot sweeps or independently randomized flag configurations."""

import argparse
import json
import platform
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from filter_llama_performance_flags import (cpu_info, filter_catalog, find_cuda_compiler,
                                            off_value, profile_prerequisite_error)


ROOT = Path(__file__).resolve().parent
SEED = 20261021
DEFAULT_COUNT = 50
DEFAULT_FRACTION = 0.5


def valid(cmake, rules):
    return not any(all(cmake.get(name) == value for name, value in rule.items())
                   for rule in rules)


def configuration_space(options, device, cuda_arch=None):
    cpu = options["cmake"]
    gpu = options["device"] if device != "cpu" else {}
    compiler = options["compiler"]
    cmake_base = {name: "OFF" for name in cpu}
    cmake_base.update({"GGML_NATIVE": "OFF", "GGML_CUDA": "ON" if device == "cuda" else "OFF",
                       "GGML_VULKAN": "ON" if device == "vulkan" else "OFF"})
    if device == "cuda":
        cmake_base["CMAKE_CUDA_ARCHITECTURES"] = cuda_arch
    runtime_base = {}
    for name, values in gpu.items():
        if name.startswith("GGML_VK_"):
            runtime_base[name] = "0"
        elif name not in {"GGML_CUDA", "GGML_VULKAN"}:
            cmake_base[name] = "OFF"
    compiler_base = {name: off_value(name) for name in compiler}
    candidates = ([("cmake", name) for name, values in cpu.items()
                   if name != "GGML_NATIVE" and "ON" in values]
                  + [("runtime", name) for name in runtime_base]
                  + [("cmake", name) for name in gpu if name not in runtime_base
                     and name not in {"GGML_CUDA", "GGML_VULKAN"}]
                  + [("compiler_flags", name) for name, values in compiler.items()
                     if any(value != off_value(name) for value in values)])
    # GGML_NATIVE changes CPU instruction selection even when individual ISA
    # switches are OFF. Include it as a regular selectable switch.
    if "GGML_NATIVE" in cpu and "ON" in cpu["GGML_NATIVE"]:
        candidates.insert(0, ("cmake", "GGML_NATIVE"))
    return cmake_base, runtime_base, compiler_base, candidates


def configuration(options, device, space, choices, rng, single=False):
    cmake_base, runtime_base, compiler_base, _ = space
    config = {"device": device, "compiler": "clang++", "cmake": cmake_base.copy(),
              "runtime": runtime_base.copy(), "compiler_flags": compiler_base.copy()}
    for kind, name in choices:
        catalog = options[{"cmake": "cmake", "runtime": "device",
                           "compiler_flags": "compiler"}[kind]]
        values = catalog.get(name, options["device"].get(name))
        off = off_value(name) if kind == "compiler_flags" else "0" if kind == "runtime" else "OFF"
        enabled = [value for value in values if value != off]
        config[kind][name] = enabled[0] if single else rng.choice(enabled)
    return config


def valid_configuration(config, options):
    return valid(config["cmake"] | config["runtime"] | config["compiler_flags"],
                 options.get("invalid_combinations", []))


def execution_plan(options, device, cuda_arch=None):
    rng = random.Random(SEED)
    space = configuration_space(options, device, cuda_arch)
    candidates = space[3]
    rng.shuffle(candidates)
    plan = []
    for candidate in candidates:
        config = configuration(options, device, space, [candidate], rng, single=True)
        if valid_configuration(config, options):
            plan.append({"phase": "sweep", "config": config})
    return plan


def random_configurations(options, device, cuda_arch=None, count=DEFAULT_COUNT,
                          fraction=DEFAULT_FRACTION):
    if count < 0 or not 0 <= fraction <= 1:
        raise ValueError("count must be non-negative and fraction must be between 0 and 1")
    rng = random.Random(SEED)
    space = configuration_space(options, device, cuda_arch)
    for _ in range(count):
        for _attempt in range(10_000):
            choices = [candidate for candidate in space[3] if rng.random() < fraction]
            config = configuration(options, device, space, choices, rng)
            if valid_configuration(config, options):
                yield {"phase": "random", "config": config}
                break
        else:
            raise ValueError("Could not sample a valid configuration after 10000 attempts; "
                             "check --fraction and the invalid-combination rules")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as destination:
        temporary = Path(destination.name)
        json.dump(data, destination, indent=2)
        destination.write("\n")
    temporary.replace(path)


def load_records(path, plan=None):
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list) or (plan is not None and len(records) > len(plan)):
        raise ValueError(f"Invalid execution list: {path}")
    for index, record in enumerate(records):
        if (not isinstance(record, dict)
                or (plan is not None and (record.get("phase") != plan[index]["phase"]
                                         or record.get("config") != plan[index]["config"]))
                or record.get("status") not in {"completed", "failed"}):
            raise ValueError(f"Execution {index + 1} differs from this plan: {path}")
    return records


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def fraction_value(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=ROOT / "llama.cpp",
                        help="Local llama.cpp checkout (default: llama.cpp beside this script)")
    common.add_argument("--model", type=Path, default=ROOT / "model.gguf",
                        help="GGUF model (default: model.gguf beside this script)")
    common.add_argument("--flags-file", type=Path, default=ROOT / "flags.json",
                        help="Flag catalog (default: flags.json beside this script)")
    common.add_argument("--device", required=True, choices=("cpu", "cuda", "vulkan"))
    common.add_argument("--cuda-arch", help="CUDA SM architecture, e.g. 86")
    common.add_argument("--build-dir", type=Path, default=ROOT / "build")
    common.add_argument("--results-dir", type=Path, default=ROOT / "bench-results")
    common.add_argument("--flags-output", type=Path)
    common.add_argument("--workers", type=positive_int, default=4)
    common.add_argument("--prompt", type=positive_int, default=512)
    common.add_argument("--generate", type=positive_int, default=128)
    common.add_argument("--repetitions", type=positive_int, default=3)
    common.add_argument("--gpu-layers", type=int, default=99)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--no-profile", action="store_true",
                        help="Build and benchmark only; do not require perf or HPCToolkit")
    commands.add_parser("sweep", parents=[common], help="One hot: enable each flag once")
    randomized = commands.add_parser("random", parents=[common], help="Toggle each flag independently")
    randomized.add_argument("--count", type=int, default=DEFAULT_COUNT)
    randomized.add_argument("--fraction", type=fraction_value, default=DEFAULT_FRACTION,
                            help="Probability of enabling each flag, from 0 to 1 (default: 0.5)")
    args = parser.parse_args()
    if (args.device == "cuda") != (args.cuda_arch is not None):
        parser.error("--cuda-arch is required for CUDA and only accepted with CUDA")
    if args.cuda_arch is not None and not all(part.isdigit() for part in args.cuda_arch.split(";")):
        parser.error("--cuda-arch must contain numeric CMake CUDA architectures")
    if args.mode == "random" and args.count < 0:
        parser.error("--count must be non-negative")
    source = args.source.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file() or not model.is_file():
        parser.error("--source must be a llama.cpp checkout and --model a GGUF file")
    results = args.results_dir.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    flags_file = args.flags_file.expanduser().resolve()
    if not flags_file.is_file():
        parser.error(f"Flag catalog not found: {flags_file}")
    with flags_file.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    arch, features = cpu_info()
    options = filter_catalog(catalog, arch, features, args.device)
    if args.flags_output:
        save_json(args.flags_output.expanduser().resolve(), options)
    plan = execution_plan(options, args.device, args.cuda_arch) if args.mode == "sweep" else None
    total = len(plan) if plan is not None else args.count
    configurations = (iter(plan) if plan is not None else
                      random_configurations(options, args.device, args.cuda_arch,
                                            args.count, args.fraction))
    # Random configurations are generated lazily, not stored in a future plan.
    record_file = results / f"{args.mode}-executions.json"
    plan_file = results / f"{args.mode}-plan.json"
    if plan is not None and plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        parser.error(f"Plan changed: use a different --results-dir or remove {plan_file}")
    records = load_records(record_file, plan)
    if len(records) > total:
        parser.error("--count is smaller than the saved execution list")
    completed = sum(record["status"] == "completed" for record in records)
    failed = len(records) - completed
    print(f"{args.mode}: {total} configurations, {completed} completed, {failed} failed "
          f"(seed {SEED})", flush=True)
    if args.dry_run:
        try:
            print(json.dumps(list(configurations), indent=2))
        except ValueError as error:
            parser.error(str(error))
        return
    if args.device == "cuda" and total and not find_cuda_compiler():
        parser.error("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc; "
                     "install the CUDA development toolkit on this host")
    if not args.no_profile:
        problem = profile_prerequisite_error(args.device)
        if problem:
            parser.error(problem)
    stages = ["benchmark"]
    if not args.no_profile:
        stages.append("CPU perf")
        if args.device == "cuda":
            stages.append("CUDA HPCToolkit")
    print("Required stages: " + ", ".join(stages), flush=True)
    system_file = results / f"host-system-{args.device}{('-' + args.cuda_arch.replace(';', '-')) if args.cuda_arch else ''}.json"
    # The model is immutable for this run list; hash it only once, not once per build.
    from hashlib import sha256
    digest = sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    model_hash = digest.hexdigest()
    settings = {"source": str(source), "model": str(model), "model_sha256": model_hash,
                "device": args.device, "cuda_arch": args.cuda_arch, "workers": args.workers,
                "prompt": args.prompt, "generate": args.generate,
                "repetitions": args.repetitions,
                "gpu_layers": 0 if args.device == "cpu" else args.gpu_layers,
                 "profile": not args.no_profile, "build_dir": str(build)}
    if args.mode == "random":
        settings.update({"fraction": args.fraction, "seed": SEED, "options": options})
    settings_file = results / f"{args.mode}-settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        parser.error(f"Run settings changed: use a different --results-dir or remove {settings_file}")
    save_json(settings_file, settings)
    if plan is not None:
        save_json(plan_file, plan)
    if not system_file.exists():
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        model_name = next((line.partition(":")[2].strip() for line in lscpu.splitlines()
                           if line.partition(":")[0].strip().lower() == "model name"), "")
        system = {"CPU Info": {"Architecture": platform.machine(),
                                "Model Name": model_name, "Flags": sorted(features)},
                  "Compilers": {"c": {"clang": "installed"},
                                "cxx": {"clang++": "installed"}},
                  "GPU Backends": {args.device.capitalize(): {"version": "enabled"}}
                  if args.device != "cpu" else {},
                  "Accelerators": {"Accelerators": {}}}
        if args.device == "cuda":
            system["Accelerators"]["Accelerators"]["nvidia"] = [
                {"gpu": "CUDA device", "compute_cap": int(args.cuda_arch.split(";")[0]) / 10}
            ]
        save_json(system_file, system)
    for index in range(total):
        try:
            item = next(configurations)
        except ValueError as error:
            parser.error(str(error))
        config = item["config"]
        if index < len(records) and (records[index].get("phase") != item["phase"]
                                     or records[index].get("config") != config):
            parser.error(f"Execution {index + 1} differs from this run: {record_file}")
        if index < len(records) and records[index]["status"] == "completed":
            directory = results / records[index]["results_directory"]
            required = ["benchmark.json", "build.json"]
            if not args.no_profile:
                required.append("cpu-metrics.json")
                if args.device == "cuda":
                    required.append("gpu-metrics.json")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                parser.error(f"Execution {index + 1} was recorded as completed but lacks "
                             f"{', '.join(missing)} in {directory}")
            continue
        previous = records[index] if index < len(records) else None
        command = [sys.executable, str(ROOT / "llama_gpu_bench-profile.py"),
                   "--source", str(source), "--model", str(model), "--build-dir", str(build),
                   "--results-dir", str(results), "--device", args.device,
                   "--flags-file", str(flags_file),
                   "--model-sha256", model_hash,
                   "--preflight-done",
                   "--system-info", str(system_file),
                   "--workers", str(args.workers), "--prompt", str(args.prompt),
                   "--generate", str(args.generate), "--repetitions", str(args.repetitions),
                   "--gpu-layers", str(0 if args.device == "cpu" else args.gpu_layers)]
        if args.no_profile:
            command.append("--no-profile")
        for name, value in config["cmake"].items():
            command.extend(["--cmake-arg", f"{name}={value}"])
        for name, value in config["runtime"].items():
            command.extend(["--runtime-arg", f"{name}={value}"])
        for value in config["compiler_flags"].values():
            command.append(f"--compiler-flag={value}")
        label = "Retrying failed execution" if previous else "Execution"
        print(f"\n{label} {index + 1}/{total}: {shlex.join(command)}", flush=True)
        start = time.monotonic()
        directory = None
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("Run directory: "):
                    directory = Path(line.removeprefix("Run directory: ").strip())
            code = process.wait()
        if directory is None or directory.parent != results:
            print(f"Execution {index + 1} exited {code} without reporting a run directory; "
                  "see the error above. No execution was recorded; rerun after fixing it.",
                  file=sys.stderr, flush=True)
            return 1
        record = {"phase": args.mode, "config": config,
                  "results_directory": directory.name,
                  "elapsed_seconds": round(time.monotonic() - start, 3),
                  "status": "failed" if code else "completed"}
        if code:
            record["exit_code"] = code
            print(f"Execution {index + 1} failed (exit {code}); build log: "
                  f"{directory / 'build.log'}", file=sys.stderr, flush=True)
        if previous:
            record["previous_attempts"] = [*previous.get("previous_attempts", []),
                                           {key: previous[key] for key in (
                                               "results_directory", "elapsed_seconds", "status", "exit_code")
                                            if key in previous}]
            records[index] = record
        else:
            records.append(record)
        save_json(record_file, records)
    failed = sum(record["status"] == "failed" for record in records)
    print(f"{args.mode} finished: {len(records) - failed}/{total} completed, "
          f"{failed} failed; execution list: {record_file}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
