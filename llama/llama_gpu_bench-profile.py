#!/usr/bin/env python3
"""Rebuild a XaaS llama.cpp source deployment and benchmark it on NVIDIA GPUs.

Requires the installed xaas package (including PyYAML), Docker, and a GGUF model.
The script can live outside the XaaS checkout; --xaas-root locates its build context.
"""

import argparse
import codecs
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import yaml

from xaas.source.system_discovery import discover_system


SPACK_REF = "v1.0.0"
SPACK_PACKAGES_REF = "v2026.03.0"
HPCTOOLKIT_VERSION = "2025.1.2"
GPU_EVENTS = ("-e", "gpu=cuda")
SCRIPTS_DIR = Path(__file__).resolve().parent


def hpctoolkit_stage(cuda_image, cuda_version):
    # Keep this stage independent of the llama.cpp source and CMake flags so
    # Docker can reuse the (expensive) Spack installation between builds.
    spec = f"hpctoolkit@{HPCTOOLKIT_VERSION}+cuda~viewer ^cuda@{cuda_version}"
    return f"""FROM docker.io/{cuda_image} AS hpctoolkit-layer
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential bzip2 ca-certificates curl file git gzip patch python3 tar unzip xz-utils \\
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch {SPACK_REF} https://github.com/spack/spack.git /opt/spack
RUN git clone --depth 1 --branch {SPACK_PACKAGES_REF} https://github.com/spack/spack-packages.git /opt/spack-packages \\
    && /opt/spack/bin/spack repo set --destination /opt/spack-packages builtin
RUN /opt/spack/bin/spack compiler find \\
    && /opt/spack/bin/spack external find cuda \\
    && /opt/spack/bin/spack config add 'packages:cuda:buildable:false' \\
    && /opt/spack/bin/spack spec '{spec}' \\
    && /opt/spack/bin/spack install '{spec}' \\
    && ln -s "$(/opt/spack/bin/spack location -i hpctoolkit@{HPCTOOLKIT_VERSION})" /opt/spack/hpctoolkit
"""


def benchmark_args(args, repetitions):
    return [
        "/source/build/bin/llama-bench", "-m", "/models/model.gguf",
        "-ngl", str(args.gpu_layers), "-p", str(args.prompt),
        "-n", str(args.generate), "-r", str(repetitions), "-o", "json",
    ]


def profile_runs(args, output_dir, run_image, model):
    gpu_dir = output_dir / "profiles" / "gpu-operations"
    cpu_dir = output_dir / "profiles" / "cpu"
    gpu_dir.mkdir(parents=True)
    cpu_dir.mkdir(parents=True)
    container_options = [
        "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp",
        "--mount", f"type=bind,source={output_dir},target=/results",
    ]
    model_mount = ["--mount", f"type=bind,source={model},target=/models/model.gguf,readonly"]
    gpu_container = ["docker", "run", "--rm", "--gpus", "all", *container_options]
    measurements = "/results/profiles/gpu-operations/measurements"
    database = "/results/profiles/gpu-operations/database"
    gpu = {
        "name": "gpu-operations", "events": list(GPU_EVENTS),
        "repetitions": args.profile_repetitions,
        "command": [*gpu_container, *model_mount, run_image, "hpcrun", "-o", measurements,
                    *GPU_EVENTS, *benchmark_args(args, args.profile_repetitions)],
        "benchmark_file": str(gpu_dir / "benchmark.json"),
        "log": str(gpu_dir / "benchmark.log"),
        "analysis_log": str(gpu_dir / "analysis.log"),
        "measurements": str(gpu_dir / "measurements"),
        "database": str(gpu_dir / "database"),
        "metrics_file": str(output_dir / "gpu-metrics.json"),
        "analysis_commands": [
            # The NVIDIA runtime supplies libcuda.so from the host. hpcstruct
            # must see it at the same path as hpcrun did during measurement.
            [*gpu_container, run_image, "hpcstruct", "-j", str(args.workers), measurements],
            [*gpu_container, run_image, "hpcprof", "-o", database, measurements],
        ],
    }
    # Docker's default seccomp policy commonly blocks perf_event_open even for
    # users allowed to collect user-space events by perf_event_paranoid.
    cpu_container = ["docker", "run", "--rm", "--gpus", "all",
                     "--security-opt", "seccomp=unconfined", "--cap-add", "PERFMON",
                     *container_options]
    cpu = {
        "name": "cpu", "tool": "perf stat", "repetitions": args.profile_repetitions,
        "benchmark_file": str(cpu_dir / "benchmark.json"),
        "log": str(cpu_dir / "benchmark.log"),
        "metrics_file": str(output_dir / "cpu-metrics.json"),
        "command": [*cpu_container, *model_mount, run_image, "python3",
                    "/results/collect_cpu_metrics.py", "-o", "/results/cpu-metrics.json",
                    "--benchmark-output", "/results/profiles/cpu/benchmark.json",
                    "--log", "/results/profiles/cpu/benchmark.log", "--",
                    *benchmark_args(args, args.profile_repetitions)],
    }
    return gpu, cpu


def cleanup_hpctoolkit_dumps(profile):
    """Keep the extracted GPU metrics, but discard the large raw HPCToolkit data."""
    if profile["status"] != "completed":
        raise ValueError("Cannot remove HPCToolkit data before GPU metrics extraction")
    metrics = json.loads(Path(profile["metrics_file"]).read_text())
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("GPU metrics output is empty; keeping HPCToolkit data")
    for name in ("measurements", "database"):
        shutil.rmtree(profile[name])
    profile["raw_hpctoolkit_dumps_removed"] = True


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def release_compiler_args(flags):
    if not flags:
        return []
    # ggml's CPU code needs non-finite values even when fast-math is requested.
    # Both -Ofast and -ffast-math enable -ffinite-math-only in GCC/Clang.
    effective_flags = list(flags)
    if any(flag in {"-Ofast", "-ffast-math"} for flag in flags):
        effective_flags.append("-fno-finite-math-only")
    release_flags = "-O3 -DNDEBUG " + " ".join(effective_flags)
    return [
        "CMAKE_C_FLAGS_RELEASE=" + release_flags,
        "CMAKE_CXX_FLAGS_RELEASE=" + release_flags,
    ]


def print_config(label, path):
    print(f"\n--- {label}: {path} ---", flush=True)
    print(path.read_text().rstrip(), flush=True)
    print(f"--- end {label} ---\n", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(directory):
    try:
        revision = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except FileNotFoundError:
        return None
    if revision.returncode:
        return None
    toplevel = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    # `git -C subdirectory` otherwise silently reports the parent checkout.
    if Path(toplevel.stdout.strip()).resolve() != directory.resolve():
        return None
    changes = subprocess.run(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, check=True,
    )
    lines = changes.stdout.splitlines()
    return {"commit": revision.stdout.strip(),
            "modified_tracked_files": any(not line.startswith("??") for line in lines),
            "untracked_files": any(line.startswith("??") for line in lines)}


def image_id(image):
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xaas-root", required=True, type=Path,
        help="Path to the xaas-containers checkout (Docker build context)",
    )
    parser.add_argument("--model", required=True, type=Path, help="Host path to a GGUF model")
    parser.add_argument(
        "--source-config", type=Path, default=Path("llama-source.yml"),
        help="Source YAML, relative to --xaas-root unless absolute",
    )
    parser.add_argument(
        "--deploy-config", type=Path, default=Path("llama-deploy.yml"),
        help="Deploy YAML, relative to --xaas-root unless absolute",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="Output directory (default: bench-results beside this script; relative paths use your shell's cwd)",
    )
    parser.add_argument("--cuda-version", default="12.8", help="CUDA toolkit minor version")
    parser.add_argument(
        "--cuda-image", default="nvidia/cuda:12.8.1-devel-ubuntu24.04",
        help="NVIDIA CUDA devel image with matching toolkit version",
    )
    parser.add_argument("--cmake-arg", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--compiler-flag", action="append", default=[], metavar="FLAG",
        help="C/C++ Release optimization flag (repeat for multiple flags)",
    )
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument(
        "--profile-repetitions", type=positive_int, default=1,
        help="llama-bench repetitions per profiling run (default: 1; timing uses --repetitions)",
    )
    parser.add_argument("--gpu-layers", type=int, default=99)
    args = parser.parse_args()

    root = args.xaas_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error("--xaas-root must be a directory")
    model = args.model.expanduser().resolve(strict=True)
    source_file = (root / args.source_config.expanduser()).resolve(strict=True)
    deploy_file = (root / args.deploy_config.expanduser()).resolve(strict=True)
    if not model.is_file() or not source_file.is_file() or not deploy_file.is_file():
        parser.error("model and both config paths must be files")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", args.cuda_version):
        parser.error("--cuda-version must be MAJOR.MINOR")
    if not re.search(rf":{re.escape(args.cuda_version)}\.[0-9]+-devel-", args.cuda_image):
        parser.error("--cuda-image must be a matching NVIDIA CUDA devel image")
    for arg in args.cmake_arg:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=[^\r\n]*", arg):
            parser.error(f"--cmake-arg must be NAME=VALUE: {arg!r}")
    for flag in args.compiler_flag:
        if not re.fullmatch(r"-[^\s]+", flag):
            parser.error(f"--compiler-flag must be a single compiler switch: {flag!r}")

    # XaaS interprets source_directory and working_directory relative to its
    # process cwd, and uses that directory as the source Docker build context.
    cwd = root
    with source_file.open() as stream:
        source = yaml.safe_load(stream)
    with deploy_file.open() as stream:
        deploy = yaml.safe_load(stream)
    if deploy.get("project_name") != "llama.cpp" or source.get("project_name") != "llama.cpp":
        parser.error("both configs must specify project_name: llama.cpp")
    expected_source = (
        f"{source['docker_repository']}:{source['project_name']}-source-{source['cpu_architecture']}"
    )
    if deploy.get("source_container") != expected_source:
        parser.error(f"deploy config source_container must be {expected_source}")
    if deploy["mode"]["mode"] != "predefined":
        parser.error("deploy config must use mode: predefined")

    # XaaS discovers the CUDA version from the host driver, which need not be the
    # toolkit version selected for a container build. Record a reproducible snapshot.
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except FileNotFoundError:
        parser.error("nvidia-smi is required on the host")
    if smi.returncode or not smi.stdout.strip():
        parser.error("nvidia-smi must see an NVIDIA GPU on the host")
    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S%fZ")
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else Path(__file__).resolve().parent / "bench-results"
    )
    output_dir = results_dir / run_id
    output_dir.mkdir(parents=True)
    source_snapshot = output_dir / "source.yml"
    source_snapshot.write_bytes(source_file.read_bytes())
    deploy_input_snapshot = output_dir / "deploy-input.yml"
    deploy_input_snapshot.write_bytes(deploy_file.read_bytes())

    system = discover_system()
    system.setdefault("GPU Backends", {})["CUDA"] = {
        "libraries": ["libcuda.so.1 (NVIDIA host driver)"],
        "version": args.cuda_version,
    }
    (output_dir / "system.json").write_text(json.dumps(system, indent=2) + "\n")
    deploy["system"]["system_discovery"] = str(output_dir / "system.json")
    deploy["mode"]["predefined_config"]["gpu_backends"] = "CUDA"
    # XaaS gets the source container from the config and generates a Dockerfile.
    effective_config = output_dir / "deploy.yml"
    effective_config.write_text(yaml.safe_dump(deploy, sort_keys=False))

    print_config("XaaS source YAML", source_snapshot)
    print_config("XaaS deploy YAML (input)", deploy_input_snapshot)
    print_config("XaaS deploy YAML (effective)", effective_config)

    log = output_dir / "build.log"

    def run_logged(label, command, dockerfile_path=None, log_path=None):
        command_line = shlex.join(str(part) for part in command)
        start = time.monotonic()
        capture_log = log_path or log
        with ExitStack() as stack:
            stream = stack.enter_context(capture_log.open("a"))
            build_input = (
                stack.enter_context(dockerfile_path.open("rb"))
                if dockerfile_path is not None else None
            )

            def report(message):
                print(message, end="", flush=True)
                stream.write(message)
                stream.flush()

            report(f"[{label}] + {command_line}\n")
            process = stack.enter_context(subprocess.Popen(
                command, cwd=cwd, stdin=build_input,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            ))
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            assert process.stdout is not None
            last_output = start
            while True:
                ready, _, _ = select.select([process.stdout], [], [], 30)
                if not ready:
                    now = time.monotonic()
                    report(
                        f"\n[{label}] still running ({int(now - start)}s elapsed; "
                        f"no output for {int(now - last_output)}s). Log: {capture_log}\n"
                    )
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                report(decoder.decode(chunk))
                last_output = time.monotonic()
            report(decoder.decode(b"", final=True))
            exit_code = process.wait()
            report(f"\n[{label}] finished in {int(time.monotonic() - start)}s (exit {exit_code})\n")
            if exit_code:
                raise subprocess.CalledProcessError(exit_code, command)
        return time.monotonic() - start

    try:
        build_started_at = datetime.now(timezone.utc)
        rebuild_start = time.monotonic()
        source_seconds = run_logged("XaaS source image", [
            "xaas", "source", "container", str(source_snapshot),
        ])
        # Use the tag reported by XaaS rather than assuming it matches the
        # configured deployment source image (some versions format enums in tags).
        source_tags = re.findall(r"Building Docker image: (\S+)", log.read_text())
        source_build_tag = source_tags[-1] if source_tags else expected_source
        if source_build_tag != expected_source:
            source_seconds += run_logged("Tag source image for deployment", [
                "docker", "tag", source_build_tag, expected_source,
            ])
        source_dockerfile = cwd / source["working_directory"] / "Dockerfile.source"
        (output_dir / "Dockerfile.source").write_bytes(source_dockerfile.read_bytes())
        source_image = expected_source
        source_image_id = image_id(source_image)
        generation_seconds = run_logged("XaaS deployment generation", [
            "xaas", "source", "deploy", "--no-build", str(effective_config),
        ])

        generated = cwd / deploy["working_directory"] / (
            "Dockerfile.deployment-" + deploy["system"]["name"]
        )
        dockerfile = generated.read_text()
        # XaaS uses its globally configured CUDA layer (often spcleth/xaas),
        # irrespective of docker_repository in the deployment config. Replace
        # only that stage with the matching official CUDA toolkit image.
        layer = f"cuda-{args.cuda_version}-layer"
        pattern = rf"^FROM docker\.io/\S+:cuda-{re.escape(args.cuda_version)} as {re.escape(layer)}$"
        dockerfile, count = re.subn(
            pattern, f"FROM docker.io/{args.cuda_image} as {layer}", dockerfile, flags=re.MULTILINE
        )
        if count != 1:
            raise RuntimeError("Could not locate exactly one XaaS CUDA layer in generated Dockerfile")

        # Inject extra -D definitions at the end of the generated llama.cpp
        # configure invocation, so explicit settings override discovered ones.
        cmake_args = [
            "CMAKE_BUILD_TYPE=Release", "CMAKE_C_FLAGS=-g", "CMAKE_CXX_FLAGS=-g",
            "CMAKE_CUDA_FLAGS=-lineinfo", *args.cmake_arg,
            *release_compiler_args(args.compiler_flag),
        ]
        cmake_flags = " ".join(shlex.quote("-D" + arg) for arg in cmake_args)
        dockerfile, count = re.subn(
            r"(cmake -B build[^\n]*?)(\s*\\\s*\n\s*&& cmake --build build)",
            lambda match: match.group(1) + " " + cmake_flags + match.group(2),
            dockerfile,
        )
        if count != 1:
            raise RuntimeError("Could not locate llama.cpp CMake command in generated Dockerfile")
        if dockerfile.count("-j $(nproc)") != 1:
            raise RuntimeError("Could not locate llama.cpp parallel build setting")
        dockerfile = dockerfile.replace("-j $(nproc)", "-j ${nproc}")
        dockerfile = (
            hpctoolkit_stage(args.cuda_image, args.cuda_version) + "\n" + dockerfile
            + "\nCOPY --from=hpctoolkit-layer /opt/spack /opt/spack\n"
            + "ENV HPCTOOLKIT=/opt/spack/hpctoolkit\n"
            + 'ENV PATH="/opt/spack/hpctoolkit/bin:${PATH}"\n'
            # Install perf in the workload image: measuring `docker run` on the
            # host only measures the Docker client, not its containerized child.
            + "RUN apt-get update && apt-get install -y --no-install-recommends "
            + "linux-tools-generic python3 && rm -rf /var/lib/apt/lists/* "
            + '&& ln -s "$(ls /usr/lib/linux-tools/*/perf | sort -V | tail -1)" '
            + "/usr/local/bin/perf\n"
        )
        build_config = output_dir / "Dockerfile"
        build_config.write_text(dockerfile)
        print_config("Generated build Dockerfile (effective)", build_config)
        configure_commands = re.findall(r"cmake -B build[^\n]*", dockerfile)
        if len(configure_commands) != 1:
            raise RuntimeError("Could not extract the effective CMake configure command")
        configure_command = configure_commands[0].rstrip(" \\")
        cmake_definitions = [
            flag for flag in shlex.split(configure_command) if flag.startswith("-D")
        ]
        # Pass the Dockerfile on stdin so it may be saved outside the build
        # context, while Docker still uses the XaaS checkout as its context.
        image = (
            f"{deploy['docker_repository']}:{deploy['project_name']}"
            f"-source-deploy-{deploy['system']['name']}"
        )
        run_image = image + "-" + run_id.lower()
        build_command = [
            "docker", "build", "--progress=plain", "-f", "-",
            "--build-arg", f"nproc={args.workers}", "-t", image, "-t", run_image, ".",
        ]
        image_build_seconds = run_logged(
            "CUDA llama.cpp image", build_command, dockerfile_path=build_config
        )
        rebuild_seconds = time.monotonic() - rebuild_start
        build_finished_at = datetime.now(timezone.utc)
        built_image_id = image_id(run_image)

        bench_command = [
            "docker", "run", "--rm", "--gpus", "all", "--mount",
            f"type=bind,source={model},target=/models/model.gguf,readonly",
            run_image, *benchmark_args(args, args.repetitions),
        ]
        gpu_profile, cpu_profile = profile_runs(args, output_dir, run_image, model)
        profile_specs = [gpu_profile, cpu_profile]
        for name in ("extract_gpu_metrics.py", "collect_cpu_metrics.py"):
            (output_dir / name).write_bytes((SCRIPTS_DIR / name).read_bytes())
        benchmark_file = output_dir / "benchmark.json"
        build_file = output_dir / f"build-{run_id}.json"
        print("Hashing the model for reproducibility...", flush=True)
        model_sha256 = sha256_file(model)
        build_record = {
            "run_timestamp_utc": run_started_at.isoformat(),
            "run_id": run_id,
            "build_started_at_utc": build_started_at.isoformat(),
            "build_finished_at_utc": build_finished_at.isoformat(),
            "benchmark_file": benchmark_file.name,
            "xaas_root": str(root),
            "source_config": str(source_snapshot),
            "deploy_input_config": str(deploy_input_snapshot),
            "effective_deploy_config": str(effective_config),
            "system_discovery": str(output_dir / "system.json"),
            "source_dockerfile": str(output_dir / "Dockerfile.source"),
            "build_dockerfile": str(build_config),
            "build_dockerfile_sha256": sha256_file(build_config),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "gpu_extractor_sha256": sha256_file(output_dir / "extract_gpu_metrics.py"),
            "cpu_collector_sha256": sha256_file(output_dir / "collect_cpu_metrics.py"),
            "xaas_git": git_state(root),
            "llama_cpp_git": git_state((root / source["source_directory"]).resolve()),
            "source_image": source_image,
            "xaas_source_build_tag": source_build_tag,
            "source_image_id": source_image_id,
            "image": run_image,
            "image_id": built_image_id,
            "cuda_image": args.cuda_image,
            "cuda_version": args.cuda_version,
            "hpctoolkit_version": HPCTOOLKIT_VERSION,
            "spack_ref": SPACK_REF,
            "spack_packages_ref": SPACK_PACKAGES_REF,
            "requested_cmake_args": args.cmake_arg,
            "requested_compiler_flags": args.compiler_flag,
            "cmake_configure_command": configure_command,
            "cmake_definitions_in_order": cmake_definitions,
            "docker_build_command": build_command,
            "docker_build_args": {"nproc": str(args.workers)},
            "build_time_seconds": {
                "source_image": round(source_seconds, 3),
                "deployment_generation": round(generation_seconds, 3),
                "cuda_image": round(image_build_seconds, 3),
                "total_rebuild": round(rebuild_seconds, 3),
            },
            "model": str(model),
            "model_sha256": model_sha256,
            "gpu_description": smi.stdout.strip(),
            "benchmark_command": bench_command,
            "benchmark_repetitions": args.repetitions,
            "profiles": profile_specs,
        }
        build_file.write_text(json.dumps(build_record, indent=2) + "\n")
        print("+", shlex.join(bench_command), flush=True)
        timing_start = time.monotonic()
        with benchmark_file.open("w") as out, (
            output_dir / "benchmark.log"
        ).open("w") as err:
            subprocess.run(bench_command, cwd=cwd, stdout=out, stderr=err, check=True)

        data = json.loads(benchmark_file.read_text())
        if not data or not all("CUDA" in row.get("backends", "") for row in data):
            raise RuntimeError("Benchmark did not report CUDA for every result; see benchmark.log")
        build_record["benchmark_wall_seconds"] = round(time.monotonic() - timing_start, 3)
        build_file.write_text(json.dumps(build_record, indent=2) + "\n")
        for profile in profile_specs:
            print("+", shlex.join(profile["command"]), flush=True)
            profile_start = time.monotonic()
            if profile is gpu_profile:
                with open(profile["benchmark_file"], "w") as out, open(profile["log"], "w") as err:
                    subprocess.run(profile["command"], cwd=cwd, stdout=out, stderr=err, check=True)
            else:
                run_logged("CPU perf", profile["command"], log_path=output_dir / "cpu-perf.log")
            profiled_data = json.loads(Path(profile["benchmark_file"]).read_text())
            if not profiled_data or not all("CUDA" in row.get("backends", "") for row in profiled_data):
                raise RuntimeError(f"{profile['name']} did not report CUDA; see {profile['log']}")
            profile["measurement_seconds"] = round(time.monotonic() - profile_start, 3)
            profile["status"] = "measured" if profile is gpu_profile else "completed"
            build_file.write_text(json.dumps(build_record, indent=2) + "\n")

        analysis_errors = []
        for profile in (gpu_profile,):
            analysis_start = time.monotonic()
            try:
                for analysis_command in profile["analysis_commands"]:
                    run_logged(
                        f"HPCToolkit {profile['name']} analysis", analysis_command,
                        log_path=Path(profile["analysis_log"]),
                    )
            except (OSError, subprocess.CalledProcessError) as error:
                profile["status"] = "analysis_failed"
                profile["analysis_error"] = str(error)
                analysis_errors.append(profile["name"])
            else:
                try:
                    run_logged("GPU metrics extraction", [
                        sys.executable, str(output_dir / "extract_gpu_metrics.py"),
                        profile["database"], "-o", profile["metrics_file"],
                    ], log_path=Path(profile["analysis_log"]))
                except (OSError, subprocess.CalledProcessError) as error:
                    profile["status"] = "extraction_failed"
                    profile["analysis_error"] = str(error)
                    analysis_errors.append(profile["name"])
                else:
                    profile["status"] = "completed"
                    print(f"{profile['name']} metrics: {profile['metrics_file']}")
            profile["analysis_seconds"] = round(time.monotonic() - analysis_start, 3)
            build_file.write_text(json.dumps(build_record, indent=2) + "\n")
        if analysis_errors:
            raise RuntimeError(
                "HPCToolkit analysis failed for " + ", ".join(analysis_errors)
                + "; see the per-profile analysis.log files"
            )
        cleanup_hpctoolkit_dumps(gpu_profile)
        build_file.write_text(json.dumps(build_record, indent=2) + "\n")
        print("Removed raw HPCToolkit measurements and database after GPU metrics extraction")
        print(f"Results: {benchmark_file}")
        print(f"Build record: {build_file}")
        print(f"CPU metrics: {cpu_profile['metrics_file']}")
        print(f"Image: {run_image} ({built_image_id})")
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"Failed: {error}\nLogs and generated files: {output_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
