"""Live active sampling of GROMACS on this machine (ml.main active-sampling --app gromacs).

The session itself is in live.py; this module builds and benchmarks one
configuration the same way as gromacs/gromacs_gpu_bench-profile.py --no-profile
(same CMake settings, mdrun arguments, warmup/repetitions and benchmark.json),
so that the results folders load as training data next to the campaign runs.

There is one benchmark, the same as in the history (see "The benchmark" below).

Work folder layout (DATA_DIR, default /data); every path can be overridden:
  gromacs/gromacs-src/            GROMACS source tree (the campaigns used 2026.4)
  gromacs/fftw-3.3.10.tar.gz      FFTW tarball; GROMACS builds it for GMX_FFT_LIBRARY=fftw3
  gromacs/inputs/water_0096.tpr   benchmark input (gromacs/prepare_input.sh)
  build/                          build directories, one per session
"""

import hashlib
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from .load import DEFAULT
from .load_gromacs import APPLICATION, CUDA_FLAGS, DERIVED_OPTIONS, HARNESS_OPTIONS
from .live import write_json
from .validation import compiler_flags


UNIT = "ns/day"
RESULTS = "gromacs"   # results folder: <workdir>/gromacs/results

# --- The benchmark --------------------------------------------------------------
# One benchmark for every machine: the GPU-resident mdrun of the GB10 and RTX 5060 Ti
# campaigns (benchmark.json, gromacs/results/*_README.md). If the team switches to
# another benchmark, change this block; the history must be measured the same way
# (load_gromacs.py reads benchmark.json).
DEVICE = "cuda"                 # CUDA builds (GMX_GPU=CUDA); "cpu" builds GMX_GPU=OFF without CUDA options
INPUT = "water_0096.tpr"        # water_GMX50_bare, 96k atoms, PME
NSTEPS = 50000
# GPU-resident: nonbonded, PME and update on the GPU. Water has no bonded interactions;
# "-bonded gpu" is then a fatal error, so bonded stays on the CPU.
MDRUN_ARGS = ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "gpu"]
WARMUP, REPETITIONS = 1, 3
# ---------------------------------------------------------------------------------
MDRUN_TIMEOUT = 1800   # seconds per mdrun
# Same for every build, not sampled (copied from gromacs_gpu_bench-profile.py)
FIXED_CMAKE = {"GMX_OPENMP": "ON", "GMX_MPI": "OFF", "GMX_HWLOC": "OFF", "GMX_NNPOT": "OFF",
               "GMX_USE_HDF5": "OFF", "GMX_USE_PLUMED": "OFF", "GMX_EXTERNAL_BLAS": "OFF",
               "GMX_EXTERNAL_LAPACK": "OFF", "GMX_BUILD_UNITTESTS": "OFF", "GMXAPI": "OFF",
               "GMX_INSTALL_NBLIB_API": "OFF", "GMX_ENABLE_CCACHE": "OFF", "BUILD_TESTING": "OFF"}
# Options that only exist for CUDA builds
CUDA_OPTIONS = {"CMAKE_CUDA_ARCHITECTURES", "GMX_CUDA_NB_SINGLE_COMPILATION_UNIT", *CUDA_FLAGS}
# gmx --version line -> build.json build_facts key (what GROMACS really built)
VERSION_FIELDS = {"GROMACS version": "gromacs_version", "Precision": "precision",
                  "SIMD instructions": "simd", "CPU FFT library": "cpu_fft",
                  "GPU FFT library": "gpu_fft", "GPU support": "gpu", "OpenMP support": "openmp",
                  "C++ compiler": "cxx_compiler", "C++ compiler flags": "cxx_flags",
                  "CUDA compiler": "cuda_compiler", "CUDA compiler flags": "cuda_flags",
                  "CUDA runtime": "cuda_runtime", "Hwloc support": "hwloc"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def sve_bits():
    """Hardware SVE vector length in bits; an ARM_SVE build must match it or mdrun aborts."""
    try:
        return str(8 * int(Path("/proc/sys/abi/sve_default_vector_length").read_text()))
    except (OSError, ValueError):
        return None


def find_cuda_compiler():
    try:
        executable = shlex.split(os.environ.get("CUDACXX", "nvcc"))[0]
    except (IndexError, ValueError):
        return None
    return shutil.which(executable)


def prepare(work, device, *, source=None, fftw=None, tpr=None, threads=None):
    """Check the work folder and toolchain once; return the benchmark context."""
    if device != DEVICE:
        raise ValueError(f"the GROMACS benchmark runs on --device {DEVICE}, got {device!r}")
    folder = work / "gromacs"
    source = Path(source or folder / "gromacs-src").expanduser().resolve()
    fftw = Path(fftw or folder / "fftw-3.3.10.tar.gz").expanduser().resolve()
    tpr = Path(tpr or folder / "inputs" / INPUT).expanduser().resolve()
    problems = []
    if not (source / "CMakeLists.txt").is_file():
        problems.append(f"GROMACS source tree missing: {source} (pass --source)")
    if not fftw.is_file():
        problems.append(f"FFTW tarball missing: {fftw} (pass --fftw)")
    if not tpr.is_file():
        problems.append(f"benchmark input missing: {tpr} (make it with gromacs/prepare_input.sh, or pass --tpr)")
    if not shutil.which("clang") or not shutil.which("clang++"):
        problems.append("clang and clang++ are required")
    if device == "cuda" and not find_cuda_compiler():
        problems.append("CUDA builds require nvcc on PATH or CUDACXX=/absolute/path/to/nvcc")
    if problems:
        raise ValueError("cannot run GROMACS live sampling:\n  " + "\n  ".join(problems))
    threads = threads or available_cores()
    if threads <= 0:
        raise ValueError("threads must be positive")
    return {"source": source, "fftw": fftw, "fftw_md5": hashlib.md5(fftw.read_bytes()).hexdigest(),
            "tpr": tpr, "tpr_sha256": sha256_file(tpr), "nsteps": NSTEPS, "threads": threads,
            "workers": threads}


def workload(context):
    # One benchmark: no workload label, as in the loaded history
    return ""


def sampling_space(values, device, hardware=None):
    """Keep learned tuning choices, but fix the backend and compiler for this session.

    CMAKE_CUDA_ARCHITECTURES follows from the GPU (load_gromacs.HARDWARE_DETERMINED).
    """
    values = {name: list(choices) for name, choices in values.items() if name not in DERIVED_OPTIONS}
    if device == "cuda":
        values["GMX_GPU"] = ["CUDA"]
    else:
        values = {name: choices for name, choices in values.items() if name not in CUDA_OPTIONS}
        values["GMX_GPU"] = ["OFF"]
    # The harness builds with clang only
    values["compiler"] = ["clang"]
    if "GMX_SIMD" in values and sve_bits() is None:
        values["GMX_SIMD"] = [simd for simd in values["GMX_SIMD"] if simd != "ARM_SVE"] or ["None"]
    return values


def split_config(config):
    """(CMake definitions, derived definitions, compiler flags, nvcc flags) of one configuration."""
    compiler = compiler_flags(APPLICATION)
    cmake, flags, cuda = {}, [], []
    cpu_build = str(config.get("GMX_GPU", "OFF")).upper() == "OFF"
    for name, value in config.items():
        value = str(value)
        if value == DEFAULT or name in DERIVED_OPTIONS or (cpu_build and name in CUDA_OPTIONS):
            continue
        if name == "compiler":
            if value != "clang":
                raise ValueError(f"live GROMACS builds use clang, not {value}")
        elif name in CUDA_FLAGS:
            cuda.append(value)
        elif name in compiler:
            flags.append(value)
        elif name in HARNESS_OPTIONS:
            raise ValueError(f"harness option {name} is not supported by live builds")
        else:
            cmake[name] = value
    # What the harness derives (load_gromacs.DERIVED_OPTIONS)
    derived = {"GMX_BUILD_OWN_FFTW": "ON" if cmake.get("GMX_FFT_LIBRARY", "fftw3").lower() == "fftw3" else "OFF"}
    if cmake.get("GMX_SIMD") == "ARM_SVE":
        bits = sve_bits()
        if bits is None:
            raise ValueError("GMX_SIMD=ARM_SVE but /proc/sys/abi/sve_default_vector_length is unreadable")
        derived["GMX_SIMD_ARM_SVE_LENGTH"] = bits
    return cmake, derived, flags, cuda


def commands(config, context, build, device="cuda"):
    """Configure and build commands of gromacs_gpu_bench-profile.py, and the mdrun command."""
    cmake, derived, flags, cuda = split_config(config)
    if (cmake.get("GMX_GPU", "OFF").upper() == "CUDA") != (device == "cuda"):
        raise ValueError("GMX_GPU must be CUDA exactly when the device is cuda")
    optimization = [] if any(flag.startswith("-O") for flag in flags) else ["-O3"]
    release_flags = " ".join(["-g", "-DNDEBUG", *optimization, *flags])
    definitions = [f"-D{name}={value}" for name, value in {**cmake, **derived}.items()]
    configure = ["cmake", "-S", str(context["source"]), "-B", str(build), "-G", "Unix Makefiles",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang",
                 "-DCMAKE_CXX_COMPILER=clang++", f"-DCMAKE_C_FLAGS_RELEASE={release_flags}",
                 f"-DCMAKE_CXX_FLAGS_RELEASE={release_flags}",
                 *(f"-D{name}={value}" for name, value in FIXED_CMAKE.items()), *definitions]
    if derived["GMX_BUILD_OWN_FFTW"] == "ON":
        configure.extend([f"-DGMX_BUILD_OWN_FFTW_URL={context['fftw']}",
                          f"-DGMX_BUILD_OWN_FFTW_MD5={context['fftw_md5']}"])
    if device == "cuda":
        configure.extend(["-DCMAKE_CUDA_HOST_COMPILER=clang++",
                          "-DCMAKE_CUDA_FLAGS=" + " ".join(["-lineinfo", *cuda])])
    compile_command = ["cmake", "--build", str(build), "--target", "gmx", "-j", str(context["workers"])]
    return configure, compile_command, cmake, derived, flags, cuda


def mdrun_command(binary, context, device):
    return [str(binary), "mdrun", "-s", str(context["tpr"]),
            "-ntmpi", "1",                       # one rank: no domain decomposition
            "-ntomp", str(context["threads"]),   # fixed OpenMP thread count
            "-pin", "on", *MDRUN_ARGS,
            "-nsteps", str(context["nsteps"]),
            "-resethway",                        # time only the second half: no startup/tuning
            "-notunepme",                        # same PME grid for every build
            "-noconfout", "-g", "md.log"]


# One row of the "R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G" table
PHASE_ROW = re.compile(r"^ (\S.*?)\s{2,}(?:(\d+)\s+(\d+)\s+(\d+)\s+)?([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_md_log(path):
    """ns/day, wall time and per-phase times from md.log; raises if the run did not finish."""
    text = Path(path).read_text(errors="replace")
    performance = re.search(r"^Performance:\s+([\d.]+)\s+([\d.]+)", text, re.M)
    wall = re.search(r"^\s+Time:\s+([\d.]+)\s+([\d.]+)", text, re.M)
    if not performance:
        raise ValueError(f"no 'Performance:' line in {path} (mdrun crashed or did not finish)")
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


def build_facts(binary):
    """What GROMACS actually built, from `gmx --version`."""
    output = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout
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
    precision = "double" if cmake.get("GMX_DOUBLE", "OFF").upper() == "ON" else "mixed"
    if not facts.get("precision", "").lower().startswith(precision):
        problems.append(f"expected {precision} precision, gmx reports {facts.get('precision')}")
    gpu = cmake.get("GMX_GPU", "OFF").upper()
    if gpu != "OFF" and gpu not in facts.get("gpu", "").upper():
        problems.append(f"GMX_GPU={cmake['GMX_GPU']} but gmx reports GPU support {facts.get('gpu')}")
    if problems:
        raise ValueError("build differs from the configuration: " + "; ".join(problems))


def benchmark(config, context, build, output, system, device="cuda", *, timings=None):
    """Build and benchmark one configuration; return the mean ns/day of the measured mdruns."""
    timings = timings if timings is not None else {}
    timings.update(configure_seconds=None, compile_seconds=None, benchmark_execution_seconds=None)

    def timed_run(key, command, **kwargs):
        start = time.perf_counter()
        try:
            return subprocess.run(command, **kwargs)
        finally:
            timings[key] = (timings.get(key) or 0.0) + time.perf_counter() - start

    record = {"run_id": f"{output.parent.name}-{output.name}", "device": device,
              "source": str(context["source"]), "tpr": str(context["tpr"]),
              "tpr_sha256": context["tpr_sha256"], "status": "running", "config": config,
              "cpu_benchmark": None}
    write_json(output / "system.json", system)
    (output / "source.yml").write_text("project_name: gromacs\n")
    try:
        configure, compile_command, cmake, derived, flags, cuda = commands(config, context, build, device)
        record.update(cmake_definitions_in_order=[f"-D{k}={v}" for k, v in {**cmake, **derived}.items()],
                      cmake=cmake, compiler_flags=flags, cuda_flags=cuda,
                      # options = the build as one flat dict, read by ml/load_gromacs.py
                      options={**cmake, **derived, "compiler": "clang",
                               **{k: str(v) for k, v in config.items() if str(v) != DEFAULT and (
                                   k in compiler_flags(APPLICATION) or (k in CUDA_FLAGS and cuda))}},
                      configure_command=configure, build_command=compile_command)
        write_json(output / "build.json", record)
        # Fresh tree each time: a reused tree can keep a stale gmx next to a new gmx_d.
        shutil.rmtree(build, ignore_errors=True)
        # The bundled FFTW build reads CFLAGS etc. from the environment: keep it clean
        environment = {name: value for name, value in os.environ.items()
                       if name not in {"CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}}
        with (output / "build.log").open("w") as log:
            for key, command in (("configure_seconds", configure), ("compile_seconds", compile_command)):
                print("+", shlex.join(command), flush=True)
                timed_run(key, command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
        binary = next((build / "bin" / name for name in ("gmx", "gmx_d") if (build / "bin" / name).is_file()), None)
        if binary is None:
            raise ValueError(f"built gmx not found in {build / 'bin'}")
        record["build_facts"] = build_facts(binary)
        check_build(record["build_facts"], cmake)
        bench = mdrun_command(binary, context, device)
        record["benchmark_command"] = bench
        write_json(output / "build.json", record)
        # mdrun overwrites instead of keeping #md.log.1# backups
        environment = {**os.environ, "GMX_MAXBACKUP": "-1"}
        runs = []
        for index in range(WARMUP + REPETITIONS):
            kind, number = ("warmup", index) if index < WARMUP else ("measured", index - WARMUP)
            directory = output / "benchmark" / f"{kind}{number}"
            directory.mkdir(parents=True)
            start = time.perf_counter()
            with (directory / "mdrun.log").open("w") as log:
                timed_run("benchmark_execution_seconds", bench, cwd=directory, env=environment, stdout=log,
                          stderr=subprocess.STDOUT, check=True, timeout=MDRUN_TIMEOUT)
            result = parse_md_log(directory / "md.log")
            result["process_s"] = round(time.perf_counter() - start, 3)
            if device == "cuda" and not re.search(r"^GPU support:\s+CUDA",
                                                  (directory / "md.log").read_text(errors="replace"), re.M):
                raise ValueError(f"mdrun did not use a CUDA build; see {directory / 'md.log'}")
            (directory / "ener.edr").unlink(missing_ok=True)
            runs.append({"kind": kind, "index": number, **result})
            print(f"   {kind} {number}: {result['value']} ns/day", flush=True)
        measured = [run["value"] for run in runs if run["kind"] == "measured"]
        score = statistics.mean(measured)
        if not math.isfinite(score) or score <= 0:
            raise ValueError("benchmark performance must be positive and finite")
        # Only publish benchmark.json after validation; the loader must not ingest failed runs.
        write_json(output / "benchmark.json", {
            "metric": "ns_per_day", "unit": UNIT, "higher_is_better": True, "aggregate": "mean",
            "mean": score, "std": statistics.stdev(measured) if len(measured) > 1 else 0.0,
            "n_runs": len(measured), "warmup_runs": WARMUP, "nsteps": context["nsteps"],
            "threads": context["threads"], "runs": [run for run in runs if run["kind"] == "measured"]})
        record.update(status="completed", performance=score)
        return score
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        record.update(status="failed", error=str(error))
        raise RuntimeError(f"build/benchmark failed; see {output}: {error}") from error
    finally:
        record.update(timings)
        write_json(output / "build.json", record)
