"""Capture container-visible hardware/toolchains in the ML system.json schema."""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command):
    """Optional discovery tools must not prevent CPU-only setup."""
    if not shutil.which(command[0]):
        return None
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              timeout=30, env={**os.environ, "LC_ALL": "C"}).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        print(f"system discovery: {' '.join(command)} failed: {error}", file=sys.stderr)
        return None


def cpu_info():
    cpu = {"Architecture": platform.machine(), "Model Name": platform.processor() or "unknown"}
    output = command_output(["lscpu", "--json"])
    if output is None:
        return cpu

    def fields(entries):
        for entry in entries:
            yield entry["field"].rstrip(":"), entry["data"]
            yield from fields(entry.get("children", []))

    values = dict(fields(json.loads(output)["lscpu"]))
    cpu["Architecture"] = values.get("Architecture", cpu["Architecture"])
    cpu["Model Name"] = values.get("Model name", cpu["Model Name"])
    flags = values.get("Flags") or values.get("Features")
    if flags:
        cpu["Flags"] = flags.lower().split()
    return cpu


def discover():
    system = {"CPU Info": cpu_info(), "Compilers": {}, "GPU Backends": {}}
    for language, names in {"c": ("gcc", "clang", "icx", "icc"),
                            "cxx": ("g++", "clang++", "icpx", "icpc")}.items():
        system["Compilers"][language] = {}
        for name in names:
            output = command_output([name, "--version"])
            # Failed probes are unknown, not evidence that compilation is impossible.
            if output:
                system["Compilers"][language][name] = output.splitlines()[0]
            elif not shutil.which(name):
                system["Compilers"][language][name] = "not found"

    # A toolkit is distinct from GPU visibility. Record both independently.
    nvcc = command_output(["nvcc", "--version"])
    if nvcc:
        system["GPU Backends"]["CUDA"] = {"version": nvcc}
    vulkan = command_output(["vulkaninfo", "--summary"])
    if vulkan:
        system["GPU Backends"]["Vulkan"] = {"version": vulkan}

    gpus = []
    if shutil.which("nvidia-smi"):
        output = command_output(["nvidia-smi", "--query-gpu=name,compute_cap",
                                 "--format=csv,noheader,nounits"])
        if output is None:
            # Preserve unknown GPU count if the driver cannot be queried.
            return system
        try:
            for name, capability in csv.reader(output.splitlines(), skipinitialspace=True):
                gpus.append({"gpu": name.strip(), "compute_cap": float(capability)})
        except ValueError:
            print("system discovery: unrecognized NVIDIA GPU query output", file=sys.stderr)
            return system
    system["Accelerators"] = {"Accelerators": {"nvidia": gpus}}
    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    system = discover()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(system, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"System information written to {args.output}")


if __name__ == "__main__":
    main()
