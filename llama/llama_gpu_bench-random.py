#!/usr/bin/env python3
"""Generate and run reproducible one-hot sweeps or random N-flag subsets."""

import argparse
import itertools
import json
import math
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
DEFAULT_FLAGS_PER_BUILD = 7
def valid(cmake, rules):
    return not any(all(cmake.get(name) == value for name, value in rule.items())
                   for rule in rules)


def execution_plan(options, device, cuda_arch=None, mode="sweep", count=DEFAULT_COUNT,
                   flag_count=DEFAULT_FLAGS_PER_BUILD):
    if count < 0 or flag_count < 1:
        raise ValueError("count must be non-negative and flags-per-build must be positive")
    if mode == "subset" and count == 0:
        return []
    rng = random.Random(SEED)
    cpu = options["cmake"]
    gpu = options["device"]
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
    candidates = ([ ("cmake", name) for name, values in cpu.items()
                    if name != "GGML_NATIVE" and "ON" in values ]
                  + [("runtime", name) for name in runtime_base]
                  + [("cmake", name) for name in gpu if name not in runtime_base
                     and name not in {"GGML_CUDA", "GGML_VULKAN"}]
                  + [("compiler_flags", name) for name, values in compiler.items()
                     if any(value != off_value(name) for value in values)])
    # GGML_NATIVE changes CPU instruction selection even when individual ISA
    # switches are OFF. Include it as a regular selectable switch.
    if "GGML_NATIVE" in cpu and "ON" in cpu["GGML_NATIVE"]:
        candidates.insert(0, ("cmake", "GGML_NATIVE"))
    rules = options.get("invalid_combinations", [])

    def configuration(choices, single=False):
        cmake = cmake_base.copy()
        runtime = runtime_base.copy()
        flags = compiler_base.copy()
        for kind, name in choices:
            values = gpu.get(name, cpu.get(name, compiler.get(name)))
            enabled_values = ([value for value in values if value != off_value(name)]
                              if kind == "compiler_flags" else
                              [value for value in values if value != ("0" if kind == "runtime" else "OFF")])
            value = enabled_values[0] if single else rng.choice(enabled_values)
            {"cmake": cmake, "runtime": runtime, "compiler_flags": flags}[kind][name] = value
        return {"device": device, "compiler": "clang++", "cmake": cmake,
                "runtime": runtime, "compiler_flags": flags}

    if mode == "sweep":
        rng.shuffle(candidates)
        choices = [(candidate,) for candidate in candidates
                   if valid(configuration_base(cmake_base, (candidate,)), rules)]
    elif mode == "subset":
        if flag_count > len(candidates):
            raise ValueError(f"Only {len(candidates)} selectable flags (requested {flag_count})")
        total = math.comb(len(candidates), flag_count)
        if count > total:
            raise ValueError(f"Only {total} distinct subsets (requested {count})")
        if total <= 100_000:
            pool = [group for group in itertools.combinations(candidates, flag_count)
                    if valid(configuration_base(cmake_base, group), rules)]
            if count > len(pool):
                raise ValueError(f"Only {len(pool)} valid subsets (requested {count})")
            rng.shuffle(pool)
            choices = pool
        else:
            # Sample ranks without materializing the combinatorial space.
            ranks = random.Random(SEED).sample(range(total), min(total, count * 20 + 1000))
            choices = []
            for rank in ranks:
                group = unrank(candidates, flag_count, rank)
                if valid(configuration_base(cmake_base, group), rules):
                    choices.append(group)
                    if len(choices) == count * 10 + 100:
                        break
    else:
        raise ValueError(f"Unknown mode: {mode}")
    plan = []
    for group in choices:
        config = configuration(group, single=mode == "sweep")
        if not valid(config["cmake"] | config["runtime"] | config["compiler_flags"], rules):
            continue
        plan.append({"phase": mode, "config": config})
        if mode == "subset" and len(plan) == count:
            break
    if mode == "subset" and len(plan) != count:
        raise ValueError(f"Only {len(plan)} valid configurations (requested {count})")
    return plan


def configuration_base(base, group):
    cmake = base.copy()
    for kind, name in group:
        if kind == "cmake":
            cmake[name] = "ON"
    return cmake


def unrank(candidates, size, rank):
    indices = []
    start = 0
    for remaining in range(size, 0, -1):
        for index in range(start, len(candidates) - remaining + 1):
            block = math.comb(len(candidates) - index - 1, remaining - 1)
            if rank < block:
                indices.append(candidates[index])
                start = index + 1
                break
            rank -= block
    return tuple(indices)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as destination:
        temporary = Path(destination.name)
        json.dump(data, destination, indent=2)
        destination.write("\n")
    temporary.replace(path)


def load_records(path, plan):
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list) or len(records) > len(plan):
        raise ValueError(f"Invalid execution list: {path}")
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or record.get("phase") != plan[index]["phase"]
                or record.get("config") != plan[index]["config"]
                or record.get("status") not in {"completed", "failed"}):
            raise ValueError(f"Execution {index + 1} differs from this plan: {path}")
    return records


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
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
    commands.add_parser("sweep", parents=[common], help="One hot: enable each flag once")
    subset = commands.add_parser("subset", parents=[common], help="Random N-flag subsets")
    subset.add_argument("--count", type=int, default=DEFAULT_COUNT)
    subset.add_argument("--flags-per-build", type=positive_int, default=DEFAULT_FLAGS_PER_BUILD)
    args = parser.parse_args()
    if (args.device == "cuda") != (args.cuda_arch is not None):
        parser.error("--cuda-arch is required for CUDA and only accepted with CUDA")
    if args.cuda_arch is not None and not all(part.isdigit() for part in args.cuda_arch.split(";")):
        parser.error("--cuda-arch must contain numeric CMake CUDA architectures")
    if args.mode == "subset" and args.count < 0:
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
    try:
        plan = execution_plan(options, args.device, args.cuda_arch, args.mode,
                              getattr(args, "count", 0), getattr(args, "flags_per_build", 1))
    except ValueError as error:
        parser.error(str(error))
    # Each mode has its own run list; a saved plan includes future executions too.
    record_file = results / f"{args.mode}-executions.json"
    plan_file = results / f"{args.mode}-plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        parser.error(f"Plan changed: use a different --results-dir or remove {plan_file}")
    records = load_records(record_file, plan)
    completed = sum(record["status"] == "completed" for record in records)
    failed = len(records) - completed
    print(f"{args.mode}: {len(plan)} configurations, {completed} completed, {failed} failed "
          f"(seed {SEED})", flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.device == "cuda" and plan and not find_cuda_compiler():
        parser.error("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc; "
                     "install the CUDA development toolkit on this host")
    problem = profile_prerequisite_error(args.device)
    if problem:
        parser.error(problem)
    print("Required stages: benchmark, CPU perf" + (", CUDA HPCToolkit" if args.device == "cuda" else ""),
          flush=True)
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
                "profile": True, "build_dir": str(build)}
    settings_file = results / f"{args.mode}-settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        parser.error(f"Run settings changed: use a different --results-dir or remove {settings_file}")
    save_json(settings_file, settings)
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
    for index in range(len(plan)):
        if index < len(records) and records[index]["status"] == "completed":
            directory = results / records[index]["results_directory"]
            required = ["benchmark.json", "cpu-metrics.json", "build.json"]
            if args.device == "cuda":
                required.append("gpu-metrics.json")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                parser.error(f"Execution {index + 1} was recorded as completed but lacks "
                             f"{', '.join(missing)} in {directory}")
            continue
        previous = records[index] if index < len(records) else None
        config = plan[index]["config"]
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
        for name, value in config["cmake"].items():
            command.extend(["--cmake-arg", f"{name}={value}"])
        for name, value in config["runtime"].items():
            command.extend(["--runtime-arg", f"{name}={value}"])
        for value in config["compiler_flags"].values():
            command.append(f"--compiler-flag={value}")
        label = "Retrying failed execution" if previous else "Execution"
        print(f"\n{label} {index + 1}/{len(plan)}: {shlex.join(command)}", flush=True)
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
    print(f"{args.mode} finished: {len(records) - failed}/{len(plan)} completed, "
          f"{failed} failed; execution list: {record_file}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
