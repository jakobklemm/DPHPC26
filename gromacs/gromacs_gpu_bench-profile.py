#!/usr/bin/env python3
"""GROMACS version of llama/llama_gpu_bench-profile.py: build GROMACS on the
host with clang and benchmark a fixed .tpr input with gmx mdrun.

Writes the same run folder as the llama script: source.yml, system.json,
build.json, benchmark.json, cpu-metrics.json, gpu-metrics.json, build.log.
benchmark.json holds ns/day (mean of the measured mdruns, higher is better).
With --cpu-tpr (CUDA builds), the same binary is also benchmarked CPU-only
(-nb cpu -pme cpu -update cpu, GPU detection off): benchmark-cpu.json.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from filter_gromacs_performance_flags import LLAMA, find_cuda_compiler, profile_prerequisite_error


ROOT = Path(__file__).resolve().parent
MDRUN_TIMEOUT = 1800   # seconds per mdrun
# Settings identical for every build, so only the build changes between runs.
# The input is GPU-resident with CUDA: nonbonded, PME and update on the GPU. (Water has no
# bonded interactions; "-bonded gpu" is then a fatal error, so bonded stays on the CPU.)
DEVICE_MDRUN_ARGS = {"cuda": ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "gpu"],
                     "cpu": ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"]}
# Same for every build (not sampled): no optional libraries that could appear on one host
# and change the build, nothing that is only needed for tests/APIs.
FIXED_CMAKE = {"GMX_OPENMP": "ON", "GMX_MPI": "OFF", "GMX_HWLOC": "OFF", "GMX_NNPOT": "OFF",
               "GMX_USE_HDF5": "OFF", "GMX_USE_PLUMED": "OFF", "GMX_EXTERNAL_BLAS": "OFF",
               "GMX_EXTERNAL_LAPACK": "OFF", "GMX_BUILD_UNITTESTS": "OFF", "GMXAPI": "OFF",
               "GMX_INSTALL_NBLIB_API": "OFF", "GMX_ENABLE_CCACHE": "OFF", "BUILD_TESTING": "OFF"}
# gmx --version line -> build.json build_facts key (what GROMACS really built)
VERSION_FIELDS = {"GROMACS version": "gromacs_version", "Precision": "precision",
                  "SIMD instructions": "simd", "CPU FFT library": "cpu_fft",
                  "GPU FFT library": "gpu_fft", "GPU support": "gpu", "OpenMP support": "openmp",
                  "C++ compiler": "cxx_compiler", "C++ compiler flags": "cxx_flags",
                  "CUDA compiler": "cuda_compiler", "CUDA compiler flags": "cuda_flags",
                  "CUDA runtime": "cuda_runtime", "Hwloc support": "hwloc"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def mdrun_args(binary, tpr, args, device=None, nsteps=None):
    return [str(binary), "mdrun", "-s", str(tpr),
            "-ntmpi", "1",                  # one rank: no domain decomposition
            "-ntomp", str(args.threads),    # fixed OpenMP thread count
            "-pin", "on",
            *DEVICE_MDRUN_ARGS[device or args.device],
            "-nsteps", str(nsteps or args.nsteps),
            "-resethway",                   # time only the second half: no startup/tuning
            "-notunepme",                   # same PME grid for every build
            "-noconfout", "-g", "md.log"]


# One row of the "R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G" table:
#  " PME 3D-FFT                1    6      10002       1.933          0.278  10.1"
PHASE_ROW = re.compile(r"^ (\S.*?)\s{2,}(?:(\d+)\s+(\d+)\s+(\d+)\s+)?([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_md_log(path):
    """ns/day, wall time and per-phase times from md.log; raises if the run did not finish."""
    text = path.read_text(errors="replace")
    performance = re.search(r"^Performance:\s+([\d.]+)\s+([\d.]+)", text, re.M)
    wall = re.search(r"^\s+Time:\s+([\d.]+)\s+([\d.]+)", text, re.M)
    if not performance:
        raise RuntimeError(f"no 'Performance:' line in {path} (mdrun crashed or did not finish)")
    phases, group = {}, "main"
    start = text.find("R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G")
    for line in (text[start:].splitlines() if start >= 0 else []):
        if line.strip().startswith("Breakdown of PME mesh"):
            group = "pme"
        if line.strip().startswith("Core t (s)"):
            break
        match = PHASE_ROW.match(line)
        if match and match.group(1) != "Total":
            phases[match.group(1).strip()] = {
                "wall_s": float(match.group(5)), "gcycles": float(match.group(6)),
                "percent": float(match.group(7)),
                "calls": int(match.group(4)) if match.group(4) else None, "group": group}
    return {"value": float(performance.group(1)),
            "core_s": float(wall.group(1)) if wall else None,
            "wall_s": float(wall.group(2)) if wall else None, "phases": phases}


def run_mdrun(command, directory, env, prefix=()):
    """One mdrun in its own folder; returns the parsed md.log."""
    directory.mkdir(parents=True)
    start = time.monotonic()
    with (directory / "mdrun.log").open("w") as log:
        subprocess.run([*prefix, *command], cwd=directory, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=MDRUN_TIMEOUT)
    result = parse_md_log(directory / "md.log")
    result["process_s"] = round(time.monotonic() - start, 3)
    (directory / "ener.edr").unlink(missing_ok=True)
    return result


def verify_mdrun(directory, device):
    """The run used the GPU for every offloaded task (llama: 'backends' check)."""
    text = (directory / "md.log").read_text(errors="replace")
    parse_md_log(directory / "md.log")
    if device == "cuda" and not re.search(r"^GPU support:\s+CUDA", text, re.M):
        raise RuntimeError(f"mdrun did not use a CUDA build; see {directory / 'md.log'}")


def verify_metrics(path, required=()):
    metrics = json.loads(path.read_text())
    if (not isinstance(metrics, dict) or not metrics
            or any(name not in metrics for name in required)):
        raise RuntimeError(f"Missing required metrics in {path}")


def build_facts(binary):
    """What GROMACS actually built, from `gmx --version`."""
    output = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                            check=True).stdout
    facts = {}
    for line in output.splitlines():
        label, _, value = line.partition(":")
        if label.strip() in VERSION_FIELDS and value.strip():
            facts.setdefault(VERSION_FIELDS[label.strip()], value.strip())
    return facts


def check_build(facts, cmake):
    """Fail if GROMACS silently built something else than requested (e.g. other SIMD)."""
    problems = []
    simd = cmake.get("GMX_SIMD")
    if simd and facts.get("simd", "").upper() != simd.upper():
        problems.append(f"GMX_SIMD={simd} but gmx reports SIMD {facts.get('simd')}")
    precision = "double" if cmake.get("GMX_DOUBLE") == "ON" else "mixed"
    if not facts.get("precision", "").lower().startswith(precision):
        problems.append(f"expected {precision} precision, gmx reports {facts.get('precision')}")
    if cmake.get("GMX_GPU", "OFF") != "OFF" and cmake["GMX_GPU"].upper() not in facts.get("gpu", "").upper():
        problems.append(f"GMX_GPU={cmake['GMX_GPU']} but gmx reports GPU support {facts.get('gpu')}")
    if problems:
        raise RuntimeError("Build differs from the configuration: " + "; ".join(problems))


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


def compiler_flag_option(flag):
    """flag group of a compiler flag, as in ml/load_llama.py: "-fno-vectorize" -> "-fvectorize"."""
    if flag.startswith("-O"):
        return "-O"
    name = flag.partition("=")[0]
    return "-f" + name.removeprefix("-fno-") if name.startswith("-fno-") else name


def cuda_flag_option(flag):
    """flag group of an nvcc flag: "-Xptxas=-O2" -> "-Xptxas -O" (as in flags.json)."""
    if flag.startswith("-Xptxas=-O"):
        return "-Xptxas -O"
    return flag.partition("=")[0]


def sve_bits():
    """Hardware SVE vector length; an ARM_SVE build must match it or mdrun aborts."""
    try:
        return str(8 * int(Path("/proc/sys/abi/sve_default_vector_length").read_text()))
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "gromacs-src")
    parser.add_argument("--tpr", type=Path, default=ROOT / "bench.tpr")
    parser.add_argument("--fftw", type=Path, default=ROOT / "fftw-3.3.10.tar.gz")
    parser.add_argument("--flags-file", type=Path, default=ROOT / "flags.json")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "bench-results")
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--tpr-sha256", help="Precomputed input hash for repeated runs")
    parser.add_argument("--fftw-md5", help="Precomputed FFTW tarball MD5")
    parser.add_argument("--preflight-done", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--system-info", type=Path, help="Host snapshot prepared once by the runner")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--compiler-flag", action="append", default=[])
    parser.add_argument("--cuda-flag", action="append", default=[],
                        help="nvcc switch appended to CMAKE_CUDA_FLAGS")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--threads", type=positive_int, default=os.cpu_count() or 1)
    parser.add_argument("--nsteps", type=positive_int, default=20000)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-tpr", type=Path,
                        help="CUDA builds: also benchmark CPU-only on this input (benchmark-cpu.json)")
    parser.add_argument("--cpu-tpr-sha256", help="Precomputed hash of --cpu-tpr")
    parser.add_argument("--cpu-nsteps", type=positive_int, help="MD steps per CPU-only mdrun")
    parser.add_argument("--skip-cpu-perf", action="store_true",
                        help="no perf stat stage (hosts where perf counters are blocked)")
    parser.add_argument("--skip-hpcstruct", action="store_true",
                        help="hpcprof without hpcstruct (gpu-metrics.json only needs run totals)")
    parser.add_argument("--no-profile", action="store_true",
                        help="benchmark only: no perf, no HPCToolkit (implies both --skip-* options)")
    parser.add_argument("--build-only", action="store_true",
                        help="stop after the build (e.g. to make the .tpr input with this gmx)")
    args = parser.parse_args()
    if args.no_profile:
        args.skip_cpu_perf = args.skip_hpcstruct = True
    source = args.source.expanduser().resolve()
    tpr = args.tpr.expanduser().resolve()
    fftw = args.fftw.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    if (not (source / "CMakeLists.txt").is_file() or not fftw.is_file()
            or not (tpr.is_file() or args.build_only)):
        parser.error("--source must be a GROMACS source tree, --tpr a .tpr file, --fftw the FFTW tarball")
    cmake = parse_assignments(parser, args.cmake_arg, "--cmake-arg")
    if (cmake.get("GMX_GPU", "OFF") == "CUDA") != (args.device == "cuda"):
        parser.error("GMX_GPU must be CUDA exactly when --device cuda")
    if any(not re.fullmatch(r"-[^\s]+", flag) for flag in args.compiler_flag + args.cuda_flag):
        parser.error("--compiler-flag/--cuda-flag must be a single compiler switch")
    if args.cuda_flag and args.device != "cuda":
        parser.error("--cuda-flag requires --device cuda")
    cpu_tpr = args.cpu_tpr.expanduser().resolve() if args.cpu_tpr else None
    if cpu_tpr and (args.device != "cuda" or not cpu_tpr.is_file() or not args.cpu_nsteps):
        parser.error("--cpu-tpr needs --device cuda, an existing .tpr and --cpu-nsteps")
    flags_file = args.flags_file.expanduser().resolve()
    if not flags_file.is_file():
        parser.error(f"Flag catalog not found: {flags_file}")
    with flags_file.open(encoding="utf-8") as source_flags:
        rules = json.load(source_flags).get("invalid_combinations", [])
    groups = {compiler_flag_option(flag): flag for flag in args.compiler_flag}
    groups.update({cuda_flag_option(flag): flag for flag in args.cuda_flag})
    if any(all({**cmake, **groups}.get(name) == value for name, value in rule.items()) for rule in rules):
        parser.error("invalid flag combination")
    if not shutil.which("clang") or not shutil.which("clang++"):
        parser.error("clang and clang++ are required")
    if args.device == "cuda" and not find_cuda_compiler():
        parser.error("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc")
    if not args.preflight_done and not args.no_profile:
        problem = profile_prerequisite_error(args.device, cpu_perf=not args.skip_cpu_perf)
        if problem:
            parser.error(problem)

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
    (output / "source.yml").write_text("project_name: gromacs\n")
    # GROMACS always builds its own FFTW from the local tarball when FFTW is the FFT library
    own_fftw = cmake.get("GMX_FFT_LIBRARY", "fftw3").lower() == "fftw3"
    fixed = {"GMX_BUILD_OWN_FFTW": "ON" if own_fftw else "OFF"}
    if cmake.get("GMX_SIMD") == "ARM_SVE":
        if not sve_bits():
            parser.error("GMX_SIMD=ARM_SVE but /proc/sys/abi/sve_default_vector_length is unreadable")
        fixed["GMX_SIMD_ARM_SVE_LENGTH"] = sve_bits()
    definitions = [*(f"-D{name}={value}" for name, value in cmake.items()),
                   *(f"-D{name}={value}" for name, value in fixed.items())]
    record = {"run_timestamp_utc": started.isoformat(), "source": str(source),
              "tpr": str(tpr), "tpr_sha256": args.tpr_sha256 or (sha256_file(tpr) if tpr.is_file() else None),
              "device": args.device,
              "flags_file": str(flags_file), "flags_sha256": sha256_file(flags_file),
              "run_id": output.name,
              "cmake_definitions_in_order": definitions,
              "cmake": cmake, "compiler_flags": args.compiler_flag, "cuda_flags": args.cuda_flag,
              # options = the build as one flat dict, read by ml/load_gromacs.py
              "options": {**cmake, **fixed, "compiler": "clang", **groups},
              "profiling": {"cpu_perf": not args.skip_cpu_perf,
                            "gpu_hpctoolkit": args.device == "cuda" and not args.no_profile,
                            "hpcstruct": args.device == "cuda" and not args.skip_hpcstruct},
              "cpu_benchmark": ({"tpr": str(cpu_tpr), "nsteps": args.cpu_nsteps,
                                 "tpr_sha256": args.cpu_tpr_sha256 or sha256_file(cpu_tpr)}
                                if cpu_tpr else None),
              "stages": {"benchmark": "pending",
                         "benchmark_cpu": "pending" if cpu_tpr else "not_applicable", "cpu_perf": "skipped" if args.skip_cpu_perf else "pending",
                         "gpu_hpctoolkit": ("not_applicable" if args.device != "cuda"
                                            else "skipped" if args.no_profile else "pending")},
              "script_sha256": sha256_file(Path(__file__).resolve())}
    record_file = output / "build.json"

    def save():
        record_file.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        optimization = [] if any(flag.startswith("-O") for flag in args.compiler_flag) else ["-O3"]
        flags = ["-g", "-DNDEBUG", *optimization, *args.compiler_flag]
        configure = ["cmake", "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang",
                     "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_C_FLAGS_RELEASE=" + " ".join(flags),
                     "-DCMAKE_CXX_FLAGS_RELEASE=" + " ".join(flags),
                     *(f"-D{name}={value}" for name, value in FIXED_CMAKE.items()), *definitions]
        if own_fftw:
            configure.extend([f"-DGMX_BUILD_OWN_FFTW_URL={fftw}",
                              f"-DGMX_BUILD_OWN_FFTW_MD5={args.fftw_md5 or hashlib.md5(fftw.read_bytes()).hexdigest()}"])
        if args.device == "cuda":
            configure.extend(["-DCMAKE_CUDA_HOST_COMPILER=clang++",
                              "-DCMAKE_CUDA_FLAGS=" + " ".join(["-lineinfo", *args.cuda_flag])])
        record["configure_command"] = configure
        build_command = ["cmake", "--build", str(build), "--target", "gmx", "-j", str(args.workers)]
        record["build_command"] = build_command
        save()
        # Fresh tree each time: a reused tree can keep a stale gmx next to a new gmx_d.
        shutil.rmtree(build, ignore_errors=True)
        # the bundled FFTW build reads CFLAGS etc. from the environment: keep it clean
        build_env = {name: value for name, value in os.environ.items()
                     if name not in {"CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}}
        start = time.monotonic()
        run_logged(configure, output / "build.log", env=build_env)
        run_logged(build_command, output / "build.log", env=build_env)
        record["build_time_seconds"] = round(time.monotonic() - start, 3)
        binary = next((build / "bin" / name for name in ("gmx", "gmx_d")
                       if (build / "bin" / name).is_file()), None)
        if binary is None:
            raise RuntimeError(f"Built gmx not found in {build / 'bin'}")
        record["build_facts"] = build_facts(binary)
        record["binary_size_bytes"] = binary.stat().st_size + sum(
            path.stat().st_size for path in (build / "lib").glob("libgromacs*")
            if path.is_file() and not path.is_symlink())
        check_build(record["build_facts"], cmake)
        if args.build_only:
            record["status"] = "built"
            save()
            print(f"Built: {binary}", flush=True)
            return 0
        env = {**os.environ, "GMX_MAXBACKUP": "-1"}   # mdrun overwrites instead of #md.log.1#
        bench = mdrun_args(binary, tpr, args)
        record["benchmark_command"] = bench
        save()
        benchmark, seconds = run_benchmark(bench, output / "benchmark", env, args, args.device,
                                           args.nsteps)
        record["benchmark_wall_seconds"] = seconds
        benchmark_file = output / "benchmark.json"
        benchmark_file.write_text(json.dumps(benchmark, indent=2) + "\n")
        record["stages"]["benchmark"] = "completed"
        save()
        if cpu_tpr:
            cpu_bench = mdrun_args(binary, cpu_tpr, args, "cpu", args.cpu_nsteps)
            record["cpu_benchmark"]["command"] = cpu_bench
            save()
            # GPU detection off: the CUDA build runs exactly like a CPU-only build
            cpu_result, seconds = run_benchmark(cpu_bench, output / "benchmark-cpu",
                                                {**env, "GMX_DISABLE_GPU_DETECTION": "1"},
                                                args, "cpu", args.cpu_nsteps)
            cpu_result["mdrun_mode"] = "cpu_only"
            record["benchmark_cpu_wall_seconds"] = seconds
            (output / "benchmark-cpu.json").write_text(json.dumps(cpu_result, indent=2) + "\n")
            record["stages"]["benchmark_cpu"] = "completed"
            save()
            print(f"CPU-only: mean {cpu_result['mean']:.3f} ns/day", flush=True)
        if not args.no_profile:
            profile(args, output, env, bench, record, save)
        if not args.skip_cpu_perf:
            print(f"CPU metrics: {output / 'cpu-metrics.json'}", flush=True)
        if args.device == "cuda" and not args.no_profile:
            print(f"GPU metrics: {output / 'gpu-metrics.json'}", flush=True)
        record["status"] = "completed"
        save()
        print(f"Results: {benchmark_file} (mean {benchmark['mean']:.3f} ns/day)", flush=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError, RuntimeError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        save()
        print(f"Failed: {error}\nLogs and generated files: {output}", file=sys.stderr, flush=True)
        return 1
    return 0


def run_benchmark(bench, bench_dir, env, args, device, nsteps):
    """warmup + measured mdruns; returns (benchmark dict, wall seconds)."""
    runs = []
    start = time.monotonic()
    for index in range(args.warmup + args.repetitions):
        kind = "warmup" if index < args.warmup else "measured"
        number = index if kind == "warmup" else index - args.warmup
        result = run_mdrun(bench, bench_dir / f"{kind}{number}", env)
        verify_mdrun(bench_dir / f"{kind}{number}", device)
        runs.append({"kind": kind, "index": number, **result})
        print(f"{bench_dir.name} {kind} {number}: {result['value']} ns/day", flush=True)
    measured = [run["value"] for run in runs if run["kind"] == "measured"]
    return ({"metric": "ns_per_day", "unit": "ns/day", "higher_is_better": True,
             "aggregate": "mean", "mean": statistics.mean(measured),
             "std": statistics.stdev(measured) if len(measured) > 1 else 0.0,
             "n_runs": len(measured), "warmup_runs": args.warmup,
             "nsteps": nsteps, "threads": args.threads,
             "runs": [run for run in runs if run["kind"] == "measured"]},
            round(time.monotonic() - start, 3))


def profile(args, output, env, bench, record, save):
    if not args.skip_cpu_perf:
        cpu_dir = output / "profiles" / "cpu"
        cpu_dir.mkdir(parents=True)
        run_logged([sys.executable, str(LLAMA / "collect_cpu_metrics.py"),
                    "-o", str(output / "cpu-metrics.json"),
                    "--benchmark-output", str(cpu_dir / "mdrun.out"),
                    "--log", str(cpu_dir / "mdrun.log"), "--", *bench],
                   output / "cpu-perf.log", env=env, cwd=cpu_dir)
        verify_mdrun(cpu_dir, args.device)
        verify_metrics(output / "cpu-metrics.json", ("cycles", "instructions"))
        record["stages"]["cpu_perf"] = "completed"
        save()
    if args.device != "cuda":
        return
    gpu_dir = output / "profiles" / "gpu-operations"
    gpu_dir.mkdir(parents=True)
    measurements = gpu_dir / "measurements"
    database = gpu_dir / "database"
    with (gpu_dir / "mdrun.log").open("w") as log:
        try:
            subprocess.run(["hpcrun", "-o", str(measurements), "-e", "gpu=cuda", *bench],
                           cwd=gpu_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=MDRUN_TIMEOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"HPCToolkit CUDA measurement failed (exit {error.returncode}); "
                f"see {gpu_dir / 'mdrun.log'}"
            ) from error
    verify_mdrun(gpu_dir, args.device)
    if not args.skip_hpcstruct:
        run_logged(["hpcstruct", "-j", str(args.workers), str(measurements)],
                   gpu_dir / "analysis.log")
    run_logged(["hpcprof", "-o", str(database), str(measurements)], gpu_dir / "analysis.log")
    run_logged([sys.executable, str(LLAMA / "extract_gpu_metrics.py"), str(database),
                "-o", str(output / "gpu-metrics.json")], gpu_dir / "analysis.log")
    verify_metrics(output / "gpu-metrics.json")
    record["stages"]["gpu_hpctoolkit"] = "completed"
    save()
    shutil.rmtree(measurements)
    shutil.rmtree(database)


if __name__ == "__main__":
    sys.exit(main())
