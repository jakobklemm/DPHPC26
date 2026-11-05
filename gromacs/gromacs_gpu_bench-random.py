#!/usr/bin/env python3
"""GROMACS version of llama/llama_gpu_bench-random.py: generate and run
reproducible one-factor sweeps or random N-flag subsets.

Each option in flags.json lists its baseline value first. A configuration sets
EVERY eligible option explicitly: baseline, except for the chosen ones.
  sweep:  the baseline build, then every non-baseline value of every option alone
  subset: --count random builds, each changes exactly --flags-per-build options
"""

import argparse
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import tempfile
import time
from hashlib import md5, sha256
from pathlib import Path

from filter_gromacs_performance_flags import (cpu_info, filter_catalog, find_cuda_compiler,
                                              profile_prerequisite_error)


ROOT = Path(__file__).resolve().parent
SEED = 20261021
DEFAULT_COUNT = 50
DEFAULT_FLAGS_PER_BUILD = 5
# CMake variable that receives --cuda-arch (GROMACS honours CMAKE_CUDA_ARCHITECTURES)
CUDA_ARCH_OPTION = "CMAKE_CUDA_ARCHITECTURES"


def valid(settings, rules):
    return not any(all(settings.get(name) == value for name, value in rule.items())
                   for rule in rules)


def execution_plan(options, device, cuda_arch=None, mode="sweep", count=DEFAULT_COUNT,
                   flag_count=DEFAULT_FLAGS_PER_BUILD):
    if count < 0 or flag_count < 1:
        raise ValueError("count must be non-negative and flags-per-build must be positive")
    if mode == "subset" and count == 0:
        return []
    rng = random.Random(SEED)
    cmake_options = {**options["cpu"], **options["device"]}
    compiler = options["compiler"]
    cuda_compiler = options.get("cuda_compiler", {})
    cmake_base = {name: values[0] for name, values in cmake_options.items()}
    cmake_base["GMX_GPU"] = "CUDA" if device == "cuda" else "OFF"
    if device == "cuda":
        cmake_base[CUDA_ARCH_OPTION] = cuda_arch
    compiler_base = {name: values[0] for name, values in compiler.items()}
    cuda_base = {name: values[0] for name, values in cuda_compiler.items()}
    groups_of = {"cmake": cmake_options, "compiler_flags": compiler, "cuda_flags": cuda_compiler}
    candidates = [(kind, name) for kind, catalog in groups_of.items()
                  for name, values in catalog.items() if len(values) > 1]
    rules = options.get("invalid_combinations", [])

    def configuration(changes):
        config = {"device": device, "compiler": "clang++", "cmake": cmake_base.copy(),
                  "compiler_flags": compiler_base.copy()}
        if cuda_base:
            config["cuda_flags"] = cuda_base.copy()
        for (kind, name), value in changes:
            config[kind][name] = value
        return config

    def alternatives(candidate):
        kind, name = candidate
        return groups_of[kind][name][1:]

    if mode == "sweep":
        rng.shuffle(candidates)
        groups = [()] + [((candidate, value),) for candidate in candidates
                         for value in alternatives(candidate)]
    elif mode == "subset":
        if flag_count > len(candidates):
            raise ValueError(f"Only {len(candidates)} selectable options (requested {flag_count})")
        groups, seen = [], set()
        for _ in range(count * 1000):
            chosen = sorted(rng.sample(candidates, flag_count))
            group = tuple((candidate, rng.choice(alternatives(candidate))) for candidate in chosen)
            if group not in seen:
                seen.add(group)
                groups.append(group)
                if len(groups) == count * 10 + 100:
                    break
    else:
        raise ValueError(f"Unknown mode: {mode}")
    plan = []
    for group in groups:
        config = configuration(group)
        if not valid(config["cmake"] | config["compiler_flags"] | config.get("cuda_flags", {}), rules):
            continue
        plan.append({"phase": mode, "config": config})
        if mode == "subset" and len(plan) == count:
            break
    if mode == "subset" and len(plan) != count:
        raise ValueError(f"Only {len(plan)} valid configurations (requested {count})")
    return plan


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


def file_hash(path, algorithm=sha256):
    digest = algorithm()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))   # respects the SLURM CPU allocation
    except AttributeError:
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=ROOT / "gromacs-src",
                        help="Unpacked GROMACS release (default: gromacs-src beside this script)")
    common.add_argument("--tpr", type=Path, default=ROOT / "bench.tpr",
                        help="Benchmark input, made once by prepare_input.sh (default: bench.tpr)")
    common.add_argument("--fftw", type=Path, default=ROOT / "fftw-3.3.10.tar.gz",
                        help="FFTW tarball GROMACS builds itself (GMX_BUILD_OWN_FFTW, no internet needed)")
    common.add_argument("--flags-file", type=Path, default=ROOT / "flags.json",
                        help="Flag catalog (default: flags.json beside this script)")
    common.add_argument("--device", required=True, choices=("cpu", "cuda"))
    common.add_argument("--cuda-arch", help="CUDA SM architecture, e.g. 121")
    common.add_argument("--build-dir", type=Path, default=ROOT / "build")
    common.add_argument("--results-dir", type=Path, default=ROOT / "bench-results")
    common.add_argument("--flags-output", type=Path)
    common.add_argument("--workers", type=positive_int, default=4, help="parallel compile jobs")
    common.add_argument("--threads", type=positive_int, default=available_cpus(),
                        help="mdrun OpenMP threads (default: all CPUs of this job)")
    common.add_argument("--nsteps", type=positive_int, default=20000, help="MD steps per mdrun")
    common.add_argument("--repetitions", type=positive_int, default=3, help="measured mdruns")
    common.add_argument("--warmup", type=int, default=1, help="unmeasured mdruns before them")
    common.add_argument("--skip-cpu-perf", action="store_true",
                        help="no perf stat stage (hosts where perf counters are blocked)")
    common.add_argument("--skip-hpcstruct", action="store_true",
                        help="hpcprof without hpcstruct (gpu-metrics.json only needs run totals)")
    common.add_argument("--no-profile", action="store_true",
                        help="benchmark only: no perf, no HPCToolkit (implies both --skip-* options)")
    common.add_argument("--dry-run", action="store_true")
    commands.add_parser("sweep", parents=[common], help="Baseline + each non-baseline value alone")
    subset = commands.add_parser("subset", parents=[common], help="Random N-option subsets")
    subset.add_argument("--count", type=int, default=DEFAULT_COUNT)
    subset.add_argument("--flags-per-build", type=positive_int, default=DEFAULT_FLAGS_PER_BUILD)
    args = parser.parse_args()
    if (args.device == "cuda") != (args.cuda_arch is not None):
        parser.error("--cuda-arch is required for CUDA and only accepted with CUDA")
    if args.cuda_arch is not None and not all(part.isdigit() for part in args.cuda_arch.split(";")):
        parser.error("--cuda-arch must contain numeric CMake CUDA architectures")
    if args.mode == "subset" and args.count < 0:
        parser.error("--count must be non-negative")
    if args.no_profile:
        args.skip_cpu_perf = args.skip_hpcstruct = True
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    source = args.source.expanduser().resolve()
    tpr = args.tpr.expanduser().resolve()
    fftw = args.fftw.expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file() or not tpr.is_file() or not fftw.is_file():
        parser.error("--source must be a GROMACS source tree, --tpr a .tpr file, --fftw the FFTW tarball")
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
        parser.error("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc")
    problem = (None if args.no_profile else
               profile_prerequisite_error(args.device, cpu_perf=not args.skip_cpu_perf))
    if problem:
        parser.error(problem)
    stages = (["benchmark"] + ([] if args.skip_cpu_perf else ["CPU perf"])
              + (["CUDA HPCToolkit"] if args.device == "cuda" and not args.no_profile else []))
    print("Required stages: " + ", ".join(stages), flush=True)
    system_file = results / f"host-system-{args.device}{('-' + args.cuda_arch.replace(';', '-')) if args.cuda_arch else ''}.json"
    # Inputs are immutable for this run list; hash them only once, not once per build.
    tpr_hash = file_hash(tpr)
    fftw_md5 = file_hash(fftw, md5)   # GROMACS checks the FFTW tarball by MD5
    settings = {"source": str(source), "tpr": str(tpr), "tpr_sha256": tpr_hash,
                "fftw": str(fftw), "fftw_md5": fftw_md5,
                "device": args.device, "cuda_arch": args.cuda_arch, "workers": args.workers,
                "threads": args.threads, "nsteps": args.nsteps,
                "repetitions": args.repetitions, "warmup": args.warmup,
                "profile": {"cpu_perf": not args.skip_cpu_perf,
                            "gpu_hpctoolkit": args.device == "cuda" and not args.no_profile,
                            "hpcstruct": args.device == "cuda" and not args.skip_hpcstruct},
                "build_dir": str(build)}
    settings_file = results / f"{args.mode}-settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        parser.error(f"Run settings changed: use a different --results-dir or remove {settings_file}")
    save_json(settings_file, settings)
    save_json(plan_file, plan)
    if not system_file.exists():
        # Same host snapshot as the llama runner, so both applications share hardware features.
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
            required = ["benchmark.json", "build.json"]
            if not args.skip_cpu_perf:
                required.append("cpu-metrics.json")
            if args.device == "cuda" and not args.no_profile:
                required.append("gpu-metrics.json")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                parser.error(f"Execution {index + 1} was recorded as completed but lacks "
                             f"{', '.join(missing)} in {directory}")
            continue
        previous = records[index] if index < len(records) else None
        config = plan[index]["config"]
        command = [sys.executable, str(ROOT / "gromacs_gpu_bench-profile.py"),
                   "--source", str(source), "--tpr", str(tpr), "--fftw", str(fftw),
                   "--build-dir", str(build), "--results-dir", str(results),
                   "--device", args.device, "--flags-file", str(flags_file),
                   "--tpr-sha256", tpr_hash, "--fftw-md5", fftw_md5,
                   "--preflight-done", "--system-info", str(system_file),
                   "--workers", str(args.workers), "--threads", str(args.threads),
                   "--nsteps", str(args.nsteps), "--repetitions", str(args.repetitions),
                   "--warmup", str(args.warmup)]
        if args.skip_cpu_perf:
            command.append("--skip-cpu-perf")
        if args.skip_hpcstruct:
            command.append("--skip-hpcstruct")
        if args.no_profile:
            command.append("--no-profile")
        for name, value in config["cmake"].items():
            command.extend(["--cmake-arg", f"{name}={value}"])
        for value in config["compiler_flags"].values():
            command.append(f"--compiler-flag={value}")
        for value in config.get("cuda_flags", {}).values():
            command.append(f"--cuda-flag={value}")
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
