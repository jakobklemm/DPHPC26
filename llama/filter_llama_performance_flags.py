"""Filter the llama performance-flag catalog for the current build host."""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AVX512 = {"avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"}
CPU_CMAKE_FLAGS = {
    "GGML_SSE42": {"sse4_2"},
    "GGML_AVX": {"avx"},
    "GGML_AVX2": {"avx2"},
    "GGML_FMA": {"fma"},
    "GGML_F16C": {"f16c"},
    "GGML_BMI2": {"bmi2"},
    "GGML_AVX_VNNI": {"avx_vnni"},
    "GGML_AVX512": AVX512,
    "GGML_AVX512_VBMI": AVX512 | {"avx512vbmi"},
    "GGML_AVX512_VNNI": AVX512 | {"avx512_vnni"},
    "GGML_AVX512_BF16": AVX512 | {"avx512_bf16"},
}
CPU_COMPILER_SWITCHES = {
    "GGML_SSE42": "-msse4.2",
    "GGML_AVX": "-mavx",
    "GGML_AVX2": "-mavx2",
    "GGML_FMA": "-mfma",
    "GGML_F16C": "-mf16c",
    "GGML_BMI2": "-mbmi2",
    "GGML_AVX_VNNI": "-mavxvnni",
    "GGML_AVX512": "-mavx512f",
    "GGML_AVX512_VBMI": "-mavx512vbmi",
    "GGML_AVX512_VNNI": "-mavx512vnni",
    "GGML_AVX512_BF16": "-mavx512bf16",
}
X86_64_V2 = {"pni", "ssse3", "sse4_1", "sse4_2", "popcnt", "cx16", "lahf_lm"}
X86_64_V3 = X86_64_V2 | {"avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "movbe", "xsave"}
X86_64_V4 = X86_64_V3 | AVX512
CPU_COMPILER_VALUES = {
    "-march=x86-64": set(),
    "-march=x86-64-v2": X86_64_V2,
    "-march=x86-64-v3": X86_64_V3,
    "-march=x86-64-v4": X86_64_V4,
    "-march=native": set(),
    "-mprefer-vector-width=128": set(),
    "-mprefer-vector-width=256": {"avx"},
    "-mprefer-vector-width=512": {"avx512f"},
}
COMPILER_OFF = {"-O": "-O0", "-march": "-march=x86-64",
                "-mprefer-vector-width": "-mprefer-vector-width=128",
                "-ffp-contract": "-ffp-contract=off"}


def off_value(name):
    return COMPILER_OFF.get(name, "-fno-" + name.removeprefix("-f"))


COMPILERS = ("clang", "clang++")


def find_cuda_compiler():
    """Find nvcc on PATH or at the location specified by CUDACXX."""
    try:
        executable = shlex.split(os.environ.get("CUDACXX", "nvcc"))[0]
    except (IndexError, ValueError):
        return None
    return shutil.which(executable)


def cpu_info() -> tuple[str, set[str]]:
    """Read the full CPU feature set."""
    output = subprocess.check_output(["lscpu"], text=True)
    features = set()
    for line in output.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"flags", "features"}:
            features.update(value.lower().split())
    if not features:
        raise RuntimeError("lscpu did not report CPU flags/features")
    return platform.machine().lower(), features


def compile_check(executable: str, flag: str, *, openmp: bool = False) -> bool:
    """Check both C and C++ compilation; link as well for OpenMP/LTO."""
    if not shutil.which(executable):
        return False
    language = "c++" if executable.endswith("++") else "c"
    source = (
        "#include <omp.h>\nint main(void) { return omp_get_max_threads() < 0; }"
        if openmp
        else "int main(void) { return 0; }"
    )
    command = [executable, "-x", language, "-", flag, "-o", "/dev/null"]
    if not openmp and flag != "-flto":
        command.insert(1, "-c")
    result = subprocess.run(command, input=source, text=True, capture_output=True, check=False)
    return result.returncode == 0


def toolchain_supports(flag: str, *, openmp: bool = False) -> bool:
    return all(compile_check(exe, flag, openmp=openmp) for exe in COMPILERS)


def filter_catalog(catalog: dict, architecture: str, features: set[str], device: str) -> dict:
    """Intersect catalog choices with detected CPU and installed compilers."""
    if not all(shutil.which(exe) for exe in COMPILERS):
        raise ValueError("clang and clang++ must be installed")
    x86 = architecture in {"x86_64", "amd64"}
    cmake = {}
    for name, values in catalog["cpu"].items():
        if name in CPU_CMAKE_FLAGS and (not x86 or not CPU_CMAKE_FLAGS[name] <= features):
            cmake[name] = ["OFF"]
            continue
        if name in CPU_COMPILER_SWITCHES and not toolchain_supports(CPU_COMPILER_SWITCHES[name]):
            cmake[name] = ["OFF"]
            continue
        if name == "GGML_OPENMP" and not toolchain_supports("-fopenmp", openmp=True):
            cmake[name] = ["OFF"]
            continue
        if name == "GGML_LTO" and not toolchain_supports("-flto"):
            cmake[name] = ["OFF"]
            continue
        cmake[name] = values
    supported = {}
    for name, values in catalog["compiler"].items():
        if not x86 and name in {"-march", "-mprefer-vector-width"}:
            continue
        eligible = [
            value
            for value in values
            if (
                value not in CPU_COMPILER_VALUES
                or (
                    x86
                    and CPU_COMPILER_VALUES[value] <= features
                    and (
                        value not in {"-march=x86-64-v3", "-march=x86-64-v4"}
                        or bool({"abm", "lzcnt"} & features)
                    )
                )
            )
            and toolchain_supports(value)
            # ggml's non-finite arithmetic requires these modes to be undone
            # by a later -fno-finite-math-only. Never advertise that false
            # experiment as a selectable configuration.
            and value not in {"-Ofast", "-ffast-math"}
            and value != off_value(name)
        ]
        if eligible:
            supported[name] = eligible
    device_flags = catalog["device"].get(device.upper(), {}) if device != "cpu" else {}
    invalid_combinations = [rule for rule in catalog.get("invalid_combinations", [])
                            if all(name in cmake or name in device_flags for name in rule)]
    return {
        "cmake": cmake,
        "device": device_flags,
        "compiler": supported,
        "invalid_combinations": invalid_combinations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "flags.json")
    parser.add_argument("--output", type=Path, default=ROOT / "llama-performance-flags-system.json")
    args = parser.parse_args()

    architecture, features = cpu_info()
    with args.input.open(encoding="utf-8") as source:
        catalog = json.load(source)
    result = filter_catalog(catalog, architecture, features, args.device)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
