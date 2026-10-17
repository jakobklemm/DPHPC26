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
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import yaml

from xaas.source.system_discovery import discover_system


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


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
    changes = subprocess.run(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    )
    return {"commit": revision.stdout.strip(), "modified_tracked_files": bool(changes.stdout.strip())}


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
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--repetitions", type=positive_int, default=3)
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

    def run_logged(label, command, dockerfile_path=None):
        command_line = shlex.join(str(part) for part in command)
        start = time.monotonic()
        with ExitStack() as stack:
            stream = stack.enter_context(log.open("a"))
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
                        f"no output for {int(now - last_output)}s). Log: {log}\n"
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
        cmake_args = ["CMAKE_BUILD_TYPE=Release", *args.cmake_arg]
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
            run_image, "/source/build/bin/llama-bench", "-m", "/models/model.gguf",
            "-ngl", str(args.gpu_layers), "-p", str(args.prompt),
            "-n", str(args.generate), "-r", str(args.repetitions), "-o", "json",
        ]
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
            "xaas_git": git_state(root),
            "llama_cpp_git": git_state((root / source["source_directory"]).resolve()),
            "source_image": source_image,
            "xaas_source_build_tag": source_build_tag,
            "source_image_id": source_image_id,
            "image": run_image,
            "image_id": built_image_id,
            "cuda_image": args.cuda_image,
            "cuda_version": args.cuda_version,
            "requested_cmake_args": args.cmake_arg,
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
        }
        build_file.write_text(json.dumps(build_record, indent=2) + "\n")
        print("+", shlex.join(bench_command), flush=True)
        with benchmark_file.open("w") as out, (
            output_dir / "benchmark.log"
        ).open("w") as err:
            subprocess.run(bench_command, cwd=cwd, stdout=out, stderr=err, check=True)

        data = json.loads(benchmark_file.read_text())
        if not data or not all("CUDA" in row.get("backends", "") for row in data):
            raise RuntimeError("Benchmark did not report CUDA for every result; see benchmark.log")
        print(f"Results: {benchmark_file}")
        print(f"Build record: {build_file}")
        print(f"Image: {run_image} ({built_image_id})")
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"Failed: {error}\nLogs and generated files: {output_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
