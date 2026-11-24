#!/usr/bin/env python3
"""Build one WarpX CPU configuration from scratch, prove its flags took effect, benchmark it."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED = {"WarpX_DIMS": "3", "WarpX_MPI": "OFF", "WarpX_OPENPMD": "OFF", "WarpX_FFT": "OFF",
         "WarpX_CCACHE": "OFF", "BUILD_TESTING": "OFF", "CMAKE_BUILD_TYPE": "Release",
         "CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++"}
# The LTO link runs the optimizer again and ignores these clang switches, so repeat them for lld.
LTO_LINK = {"-fno-vectorize": ["-vectorize-loops=false"],
            "-fno-slp-vectorize": ["-vectorize-slp=false"],
            "-fno-unroll-loops": ["-unroll-runtime=false", "-unroll-allow-partial=false",
                                  "-unroll-full-max-count=0"]}


def run_logged(command, log):
    print("+", shlex.join(command), flush=True)
    with log.open("a") as out:
        out.write("+ " + shlex.join(command) + "\n")
        out.flush()
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)


def object_hash(path):
    text = subprocess.run(["llvm-objdump", "-dr", "--no-show-raw-insn", str(path)],
                          capture_output=True, text=True, check=True).stdout
    lines = [l for l in text.splitlines() if "file format" not in l]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest() if len(lines) > 3 else ""


def code_hash(build, lto):
    """Hash the disassembly of all objects. Not the linked binary: it embeds build dates and flag
    strings that shift addresses, so it differs even for identical code. Under LTO the objects
    hold bitcode, so use the native objects the linker generated."""
    files = list((build / "lto").glob("llvmcache-*")) if lto else list(build.rglob("*.o"))
    files = [f for f in files if "buildInfo" not in f.name]
    with ThreadPoolExecutor() as pool:
        # A sorted list of hashes: independent of file naming.
        hashes = sorted(h for h in pool.map(object_hash, files) if h)
    if not hashes:
        raise RuntimeError("no code found to hash")
    return hashlib.sha256("".join(hashes).encode()).hexdigest(), len(hashes)


def verify_build(build, cmake, flags, record):
    problems = []
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([A-Za-z_0-9]+):([A-Z]+)=(.*)", line)
        if match:
            cache[match[1]] = (match[2], match[3])
    for name, value in cmake.items():
        kind, actual = cache.get(name, ("missing", None))
        # UNINITIALIZED: no option() declares it, so CMake ignored it (BUILD_SHARED_LIBS is a
        # CMake built-in and is checked on the linked binary instead).
        if actual != value or (kind == "UNINITIALIZED" and name != "BUILD_SHARED_LIBS"):
            problems.append(f"{name}={value} but CMake has {actual} ({kind})")
    text = (build / "CMakeFiles/rules.ninja").read_text() + (build / "build.ninja").read_text()
    config = next(build.rglob("AMReX_Config_3D.H"), None)
    defines = set(re.findall(r"^#define (\w+)", config.read_text(), re.M)) if config else set()
    expect = {"AMREX_USE_OMP": cmake["WarpX_COMPUTE"] == "OMP",
              "AMREX_USE_SIMD": cmake["WarpX_SIMD"] == "ON",
              "AMREX_USE_EB": cmake["WarpX_EB"] == "ON",
              "AMREX_USE_FLOAT": cmake["WarpX_PRECISION"] == "SINGLE",
              "AMREX_SINGLE_PRECISION_PARTICLES": cmake["WarpX_PARTICLE_PRECISION"] == "SINGLE",
              "AMREX_USE_FLATTEN_FOR": cmake["AMReX_FLATTEN_FOR"] == "ON",
              "AMREX_ARRAY4_64BIT_INDEX": cmake["AMReX_ARRAY4_64BIT_INDEX"] == "ON"}
    for name, wanted in expect.items():
        if (name in defines) != wanted:
            problems.append(f"{name} defined={name in defines}, expected {wanted}")
    commands = json.loads((build / "compile_commands.json").read_text())
    commands = [c["command"] for c in commands if c["file"].endswith((".cpp", ".cxx"))]
    tail = " ".join(flags)
    for command in commands:
        before_output = command.split(" -o ")[0]
        if not before_output.endswith(tail):
            problems.append("catalog flags are not last in: " + command[-300:])
            break
    # Whole-program settings: check on the WarpX sources, where the effect is wanted.
    warpx = [c for c in commands if "/opt/src/warpx/Source" in c or "unity_0" in c]
    for switch, wanted in {"-flto": cmake["WarpX_IPO"] == "ON",
                           "-ffast-math": cmake["WarpX_FASTMATH"] == "ON",
                           "-fno-math-errno": cmake["AMReX_MATH_ERRNO"] == "OFF"}.items():
        if not warpx or not all((switch in c) == wanted for c in warpx):
            problems.append(f"{switch} on all WarpX sources should be {wanted}")
    if any("unity_0" in c for c in warpx) != (cmake["WarpX_UNITY_BUILD"] == "ON"):
        problems.append("unity build setting did not take effect")
    record["compile_commands_checked"] = len(commands)
    record["sample_compile_command"] = warpx[0] if warpx else None
    if problems:
        raise RuntimeError("flags did not take effect: " + "; ".join(problems))


def parse_run(text):
    step = re.findall(r"^STEP (\d+) ends", text, re.M)
    evolve = re.search(r"^Total Time\s*:\s*([\d.eE+-]+)", text, re.M)
    region = re.search(r"^WarpX::Evolve\(\)\s+\d+\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                       text.split("Incl. Min")[-1], re.M)
    if not step or not evolve or not region:
        raise RuntimeError("could not parse WarpX output")
    return int(step[-1]), float(evolve[1]), float(region[2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--system-info", type=Path, required=True)
    p.add_argument("--config", required=True, help="JSON configuration")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--repetitions", type=int, required=True)
    args = p.parse_args()
    config = json.loads(args.config)
    cmake = config["cmake"]
    flags = [v for v in config["compiler_flags"].values() if v != "default"]
    started = datetime.now(timezone.utc)
    output = args.results_dir.resolve() / started.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    print(f"Run directory: {output}", flush=True)
    shutil.copy(args.system_info, output / "system.json")
    (output / "source.yml").write_text("project_name: warpx\n")
    record = {"run_timestamp_utc": started.isoformat(), "run_id": output.name, "device": "cpu",
              "source_git": {n: subprocess.check_output(
                  ["git", "-C", f"/opt/src/{n}", "rev-parse", "HEAD"], text=True).strip()
                  for n in ("warpx", "amrex", "picsar", "vir-simd")},
              "cmake": cmake, "compiler_flags": flags, "stages": {"benchmark": "pending"},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    build = args.build_dir.resolve()
    try:
        # Always from scratch: WarpX FORCE-caches options (ABLASTR_FASTMATH, PIC), which would
        # otherwise leak from one configuration into the next.
        if build.exists():
            shutil.rmtree(build)
        tail = " ".join(flags)
        link = ["-fuse-ld=lld"]
        if cmake["WarpX_IPO"] == "ON":
            # Fresh per build: only keeps the native objects of every link, for code_hash().
            link.append(f"-Wl,--thinlto-cache-dir={build}/lto,--thinlto-cache-policy=prune_after=0s:cache_size=0%")
            link += [f"-Wl,-mllvm,{o}" for f in flags for o in LTO_LINK.get(f, [])]
        link = " ".join(link)
        configure = [
            "cmake", "-G", "Ninja", "-S", "/opt/src/warpx", "-B", str(build),
            "-DWarpX_amrex_src=/opt/src/amrex", "-DWarpX_picsar_src=/opt/src/picsar",
            *[f"-D{n}={v}" for n, v in (FIXED | cmake).items()],
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            # In the release flags so the link step sees them, and again LAST on every compile so
            # that nothing a dependency adds (-ffast-math, ...) can override the catalog's choice.
            f"-DCMAKE_C_FLAGS_RELEASE=-DNDEBUG {tail}", f"-DCMAKE_CXX_FLAGS_RELEASE=-DNDEBUG {tail}",
            "-DCMAKE_CXX_COMPILE_OBJECT=<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> "
            f"{tail} -o <OBJECT> -c <SOURCE>",
            f"-DCMAKE_EXE_LINKER_FLAGS={link}", f"-DCMAKE_SHARED_LINKER_FLAGS={link}"]
        command = ["cmake", "--build", str(build), "--target", "app_3d", "-j", str(args.workers)]
        record.update(configure_command=configure, build_command=command)
        save()
        start = time.monotonic()
        run_logged(configure, output / "build.log")
        run_logged(command, output / "build.log")
        record["build_time_seconds"] = round(time.monotonic() - start, 3)
        verify_build(build, cmake, flags, record)
        binary = (build / "bin/warpx.3d").resolve(strict=True)
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        own = sorted({Path(m) for m in re.findall(r"=> (\S+)", linked) if m.startswith(str(build))})
        if bool(own) != (cmake["BUILD_SHARED_LIBS"] == "ON"):
            raise RuntimeError("flags did not take effect: BUILD_SHARED_LIBS")
        if ("libomp" in linked) != (cmake["WarpX_COMPUTE"] == "OMP"):
            raise RuntimeError("flags did not take effect: WarpX_COMPUTE")
        code, objects = code_hash(build, cmake["WarpX_IPO"] == "ON")
        record["binary"] = {"name": binary.name, "linked_build_libraries": [f.name for f in own],
                            "size_bytes": sum(f.stat().st_size for f in [binary, *own]),
                            "code_sha256": code, "hashed_objects": objects}
        threads = args.threads if cmake["WarpX_COMPUTE"] == "OMP" else 1
        env = os.environ | {"OMP_NUM_THREADS": str(threads), "OMP_PLACES": "cores",
                            "OMP_PROC_BIND": "close"}
        bench = [str(binary), str(ROOT / "inputs_bench_3d"), f"max_step={args.steps}"]
        record.update(benchmark_command=bench, threads=threads)
        save()
        work = build / "run"
        work.mkdir()
        samples, totals = [], []
        start = time.monotonic()
        with (output / "benchmark.log").open("w") as log:
            for _ in range(args.repetitions):
                done = subprocess.run(bench, cwd=work, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
                log.write(done.stdout)
                log.flush()
                if done.returncode:
                    raise RuntimeError(f"warpx exited {done.returncode}; see benchmark.log")
                steps, total, evolve = parse_run(done.stdout)
                if steps != args.steps:
                    raise RuntimeError(f"warpx stopped after {steps} steps")
                samples.append(evolve)
                totals.append(total)
        particles = 64 ** 3 * 8
        (output / "benchmark.json").write_text(json.dumps([{
            "backends": "CPU", "n_threads": threads, "steps": args.steps, "particles": particles,
            "samples_evolve_seconds": samples, "samples_total_seconds": totals,
            "avg_evolve_seconds": statistics.mean(samples),
            "stddev_evolve_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "avg_particle_steps_per_second": particles * args.steps / statistics.mean(samples),
        }], indent=2) + "\n")
        record["benchmark_wall_seconds"] = round(time.monotonic() - start, 3)
        record["stages"]["benchmark"] = record["status"] = "completed"
        save()
        print(f"Results: {output / 'benchmark.json'}", flush=True)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        record.update(status="failed", error=str(error))
        save()
        print(f"Failed: {error}\nLogs: {output}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
