#!/usr/bin/env python3
"""Build one MILC CPU configuration from scratch, prove its flags took effect, benchmark it."""

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
# The application is hardcoded: ks_imp_rhmc / su3_rhmd_hisq.
SUBDIR, TARGET = "ks_imp_rhmc", "su3_rhmd_hisq"
FIXED = {"SUBDIR": SUBDIR, "MPP": "OFF", "CMAKE_BUILD_TYPE": "Release", "CMAKE_C_COMPILER": "clang",
         "CMAKE_CXX_COMPILER": "clang++", "CMAKE_AR": "/usr/bin/llvm-ar",
         "CMAKE_RANLIB": "/usr/bin/llvm-ranlib"}
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
    """Hash the disassembly of all objects: the linked binary embeds strings that shift addresses.
    Under LTO the objects hold bitcode, so use the native code the linker generated. show_generic*
    only prints the option names."""
    files = list((build / "lto").glob("obj*")) if lto else list(build.rglob("*.o"))
    files = [f for f in files if "show_generic" not in f.name]
    with ThreadPoolExecutor() as pool:
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
    for name, value in ({"SUBDIR": SUBDIR, "MPP": "OFF"} | cmake).items():
        kind, actual = cache.get(name, ("missing", None))
        if actual != value or kind == "UNINITIALIZED":
            problems.append(f"{name}={value} but CMake has {actual} ({kind})")
    text = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", TARGET], text=True)
    commands = [l for l in text.splitlines() if " -c " in l]
    sources = [c for c in commands if f"/{TARGET}.dir/" in c]
    libs = [c for c in commands if "/libraries/" in c.split(" -c ")[-1] and "crc32" not in c]
    tail = " ".join(flags)
    for command in commands:
        if not command.split(" -o ")[0].endswith(tail):
            problems.append("catalog flags are not last in: " + command[-300:])
            break
    words = [set(c.split()) for c in sources]
    expect = {f"-DMILC_PRECISION={p}": cmake["PRECISION"] == p for p in ("1", "2")}
    expect |= {"-DOMP": cmake["OMP"] == "ON", "-DC_INLINE": cmake["INLINE_C"] == "ON",
               "-DC_GLOBAL_INLINE": cmake["INLINE_C_GLOBAL"] == "ON",
               "-DSSE_INLINE": cmake["INLINE_SSE"] == "ON",
               "-DSSE_GLOBAL_INLINE": cmake["INLINE_SSE_GLOBAL"] == "ON",
               "-DHAVE_QUDA": cmake["WANTQUDA"] == "ON", "-DKS_MULTICG=" + cmake["KS_MULTICG"]: True}
    for define, wanted in expect.items():
        if not words or not all((define in w) == wanted for w in words):
            problems.append(f"{define} on all sources should be {wanted}")
    if any(sum(x.startswith("-DKS_MULTICG=") for x in w) != 1 for w in words):
        problems.append("KS_MULTICG is defined more than once")
    if not libs or not all(f"-DMILC_PRECISION={cmake['PRECISION']}" in c for c in libs):
        problems.append("libraries have the wrong precision")
    # The last switch of each kind wins; nothing but the catalog may set one after it.
    record["compile_commands_checked"] = len(commands)
    record["sample_compile_command"] = sources[0] if sources else None
    if problems:
        raise RuntimeError("flags did not take effect: " + "; ".join(problems))


def parse_run(text):
    seconds = re.search(r"^Aggregate time to do one trajectory\s+([\d.eE+-]+)", text, re.M)
    total = re.search(r"^Time = ([\d.eE+-]+) seconds", text, re.M)
    iters = re.search(r"^total_iters = (\d+)", text, re.M)
    plaq = re.findall(r"^PLAQ:\s+(\S+)\s+(\S+)", text, re.M)
    if not (seconds and total and iters and plaq and "RUNNING COMPLETED" in text):
        raise RuntimeError("could not parse MILC output")
    if "NOT converged" in text:
        raise RuntimeError("the CG solver did not converge")
    return float(seconds[1]), float(total[1]), int(iters[1]), [float(x) for x in plaq[-1]]


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
    (output / "source.yml").write_text("project_name: milc\n")
    record = {"run_timestamp_utc": started.isoformat(), "run_id": output.name, "device": "cpu",
              "source_git": {n: subprocess.check_output(
                  ["git", "-C", f"/opt/src/{n}", "rev-parse", "HEAD"], text=True).strip()
                  for n in ("milc",)},
              "cmake": cmake, "compiler_flags": flags, "stages": {"benchmark": "pending"},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    build = args.build_dir.resolve()
    try:
        # Always from scratch, without ccache: nothing can leak from one configuration to the next.
        if build.exists():
            shutil.rmtree(build)
        tail = " ".join(flags)
        link = ["-fuse-ld=lld"]
        lto = config["compiler_flags"]["-flto"] != "-fno-lto"
        if lto:
            # Keeps the native objects of the link, for code_hash().
            link.append(f"-Wl,--lto-obj-path={build}/lto/obj")
            link += [f"-Wl,-mllvm,{o}" for f in flags for o in LTO_LINK.get(f, [])]
        link = " ".join(link)
        (build / "lto").mkdir(parents=True)
        configure = [
            "cmake", "-G", "Ninja", "-S", "/opt/src/milc", "-B", str(build),
            *[f"-D{n}={v}" for n, v in (FIXED | cmake).items()],
            # In the release flags so the link step sees them, and again LAST on every compile.
            f"-DCMAKE_C_FLAGS_RELEASE=-DNDEBUG {tail}", f"-DCMAKE_CXX_FLAGS_RELEASE=-DNDEBUG {tail}",
            *[f"-DCMAKE_{l}_COMPILE_OBJECT=<CMAKE_{l}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> "
              f"{tail} -o <OBJECT> -c <SOURCE>" for l in ("C", "CXX")],
            f"-DCMAKE_EXE_LINKER_FLAGS={link}"]
        command = ["cmake", "--build", str(build), "--target", TARGET, "-j", str(args.workers)]
        record.update(configure_command=configure, build_command=command)
        save()
        start = time.monotonic()
        run_logged(configure, output / "build.log")
        run_logged(command, output / "build.log")
        record["build_time_seconds"] = round(time.monotonic() - start, 3)
        verify_build(build, cmake, flags, record)
        binary = (build / SUBDIR / TARGET).resolve(strict=True)
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        if ("libomp" in linked) != (cmake["OMP"] == "ON"):
            raise RuntimeError("flags did not take effect: OMP")
        code, objects = code_hash(build, lto)
        record["binary"] = {"name": binary.name, "size_bytes": binary.stat().st_size,
                            "code_sha256": code, "hashed_objects": objects}
        threads = args.threads if cmake["OMP"] == "ON" else 1
        env = os.environ | {"OMP_NUM_THREADS": str(threads), "OMP_PLACES": "cores",
                            "OMP_PROC_BIND": "close"}
        work = build / "run"
        work.mkdir()
        inputs = (ROOT / "inputs_bench.template").read_text()
        for key, value in {"@WARMS@": "0", "@STEPS@": str(args.steps), "@SAVE@": "forget",
                           "@LOAD@": "reload_serial /opt/data/lattice.l8888"}.items():
            inputs = inputs.replace(key, value)
        (work / "bench.in").write_text(inputs)
        bench = [str(binary), "bench.in"]
        record.update(benchmark_command=bench, threads=threads)
        save()
        samples, totals, iterations, plaquettes = [], [], [], []
        start = time.monotonic()
        with (output / "benchmark.log").open("w") as log:
            for _ in range(args.repetitions):
                done = subprocess.run(bench, cwd=work, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace")
                log.write(done.stdout)
                log.flush()
                if done.returncode:
                    raise RuntimeError(f"milc exited {done.returncode}; see benchmark.log")
                seconds, total, iters, plaq = parse_run(done.stdout)
                samples.append(seconds)
                totals.append(total)
                iterations.append(iters)
                plaquettes.append(plaq)
        (output / "benchmark.json").write_text(json.dumps([{
            "backends": "CPU", "n_threads": threads, "steps": args.steps, "lattice": "8^4",
            "samples_trajectory_seconds": samples, "samples_total_seconds": totals,
            "cg_iterations": iterations, "plaquettes": plaquettes,
            "avg_trajectory_seconds": statistics.mean(samples),
            "stddev_trajectory_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "avg_trajectories_per_second": 1 / statistics.mean(samples),
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
