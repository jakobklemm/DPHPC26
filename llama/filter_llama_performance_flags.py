"""Filter the llama performance-flag catalog for the current build host."""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
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
ARM = {"aarch64", "arm64"}
# ARM instruction levels as -mcpu values, with the required features in Linux
# /proc/cpuinfo spelling (macOS sysctl names are mapped in cpu_info). The
# baseline -mcpu=generic is plain ARMv8-A NEON, like -march=x86-64 on x86.
# -mcpu, not -march: clang ignores -mcpu=native once any -march is given, so a
# -march baseline would silently disable GGML_NATIVE (ggml appends -mcpu=native
# after our flags, and the last -mcpu wins).
ARM_COMPILER_VALUES = {
    "-mcpu=native": set(),
    "-mcpu=generic+dotprod": {"asimddp"},
    "-mcpu=generic+dotprod+fp16": {"asimddp", "asimdhp"},
    "-mcpu=generic+dotprod+fp16+i8mm": {"asimddp", "asimdhp", "i8mm"},
    "-mcpu=generic+dotprod+fp16+i8mm+sve": {"asimddp", "asimdhp", "i8mm", "sve"},
}
# CMake options that only exist for ARM hosts; left out entirely elsewhere so
# x86 plans and configurations stay unchanged.
ARM_CMAKE_FLAGS = {"GGML_CPU_KLEIDIAI"}
# macOS reports CPU features through sysctl hw.optional.*; map them to the
# Linux names used above and by lscpu on ARM Linux hosts.
DARWIN_FEATURES = {
    "neon": ("fp", "asimd"),
    "arm.FEAT_FP16": ("fphp", "asimdhp"),
    "arm.FEAT_DotProd": ("asimddp",),
    "arm.FEAT_I8MM": ("i8mm",),
    "arm.FEAT_BF16": ("bf16",),
    "arm.FEAT_FHM": ("asimdfhm",),
    "arm.FEAT_RDM": ("asimdrdm",),
    "arm.FEAT_LSE": ("atomics",),
    "arm.FEAT_CRC32": ("crc32",),
    "arm.FEAT_AES": ("aes",),
    "arm.FEAT_SHA256": ("sha2",),
    "arm.FEAT_SHA3": ("sha3",),
    "arm.FEAT_SME": ("sme",),
    "arm.FEAT_SME2": ("sme2",),
}
COMPILER_OFF = {"-O": "-O3", "-march": "-march=x86-64", "-mcpu": "-mcpu=generic",
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


def profile_prerequisite_error(device):
    """Return the first reason a complete host-side profile cannot run."""
    if device == "vulkan":
        return "Vulkan GPU profiling is not supported by the CUDA HPCToolkit collector"
    required = ["perf", *(["hpcrun", "hpcstruct", "hpcprof"] if device == "cuda" else [])]
    missing = [tool for tool in required if not shutil.which(tool)]
    if missing:
        return "Profiling requires host tools: " + ", ".join(missing)

    from collect_cpu_metrics import parse_stat

    try:
        probe = subprocess.run(
            ["perf", "stat", "--no-big-num", "-x", ";", "-e", "cycles:u,instructions:u", "--", "true"],
            capture_output=True, text=True,
        )
    except OSError as error:
        return f"Cannot run perf stat: {error}"
    counters = parse_stat(probe.stderr)
    if probe.returncode or not {"cycles", "instructions"} <= counters.keys():
        return ("Cannot access CPU counters (perf stat failed): "
                + (probe.stderr.strip() or "no cycles/instructions counted"))
    if device == "cuda":
        try:
            with tempfile.TemporaryDirectory(prefix="llama-hpctoolkit-check-") as temporary:
                gpu_probe = subprocess.run(
                    ["hpcrun", "-o", str(Path(temporary) / "measurements"), "-e", "gpu=cuda",
                     shutil.which("true") or "true"], capture_output=True, text=True,
                )
        except OSError as error:
            return f"Cannot run HPCToolkit CUDA probe: {error}"
        if gpu_probe.returncode:
            detail = (gpu_probe.stderr or gpu_probe.stdout).strip()
            if "not compiled with NVIDIA CUDA support" in detail:
                return ("HPCToolkit was built without CUDA support; install and load "
                        "a +cuda build (e.g. spack install hpctoolkit+cuda), then retry")
            return f"HPCToolkit CUDA probe failed: {detail or f'exit {gpu_probe.returncode}'}"
    return None


def cpu_info() -> tuple[str, set[str]]:
    """Read the full CPU feature set."""
    if platform.system() == "Darwin":
        output = subprocess.check_output(["sysctl", "hw.optional"], text=True)
        features = set()
        for line in output.splitlines():
            key, _, value = line.partition(":")
            names = DARWIN_FEATURES.get(key.strip().removeprefix("hw.optional."))
            if names and value.strip() == "1":
                features.update(names)
        if not features:
            raise RuntimeError("sysctl hw.optional did not report CPU features")
        return platform.machine().lower(), features
    output = subprocess.check_output(["lscpu"], text=True)
    features = set()
    for line in output.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"flags", "features"}:
            features.update(value.lower().split())
    if not features:
        raise RuntimeError("lscpu did not report CPU flags/features")
    return platform.machine().lower(), features


def cpu_model_name() -> str:
    """CPU model string for the host snapshot (lscpu does not exist on macOS)."""
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
    return next((line.partition(":")[2].strip() for line in lscpu.splitlines()
                 if line.partition(":")[0].strip().lower() == "model name"), "")


def openmp_flags() -> list[str]:
    """Apple clang has no -fopenmp driver flag: use libomp from OpenMP_ROOT
    (e.g. Homebrew's), the same root CMake's FindOpenMP uses for GGML_OPENMP."""
    root = os.environ.get("OpenMP_ROOT")
    if platform.system() == "Darwin" and root:
        return ["-Xclang", "-fopenmp", f"-I{root}/include", f"-L{root}/lib", "-lomp"]
    return ["-fopenmp"]


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
    switches = openmp_flags() if openmp and flag == "-fopenmp" else [flag]
    command = [executable, "-x", language, "-", *switches, "-o", "/dev/null"]
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
    arm = architecture in ARM
    cmake = {}
    for name, values in catalog["cpu"].items():
        if name in ARM_CMAKE_FLAGS and not arm:
            continue
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
        if not arm and name == "-mcpu":
            continue
        eligible = [
            value
            for value in values
            if (value not in ARM_COMPILER_VALUES or ARM_COMPILER_VALUES[value] <= features)
            and (
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
            and value not in {"-O0", "-Ofast", "-ffast-math"}
            and value != off_value(name)
        ]
        if eligible:
            supported[name] = eligible
    device_flags = catalog["device"].get(device.upper(), {}) if device != "cpu" else {}
    invalid_combinations = [rule for rule in catalog.get("invalid_combinations", [])
                             if all(name in cmake or name in device_flags or name in supported
                                    for name in rule)]
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
