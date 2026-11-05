"""Filter the GROMACS performance-flag catalog for the current build host.

Same role as llama/filter_llama_performance_flags.py. Catalog convention
(gromacs/flags.json): for every option the FIRST value is the baseline, i.e.
what a normal GROMACS Release build uses; the other values are alternatives.
"""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LLAMA = ROOT.parent / "llama"   # shared profilers: collect_cpu_metrics.py, extract_gpu_metrics.py
sys.path.insert(0, str(LLAMA))

COMPILERS = ("clang", "clang++")
X86 = {"x86_64", "amd64"}
ARM = {"aarch64", "arm64"}
AVX512 = {"avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"}
# GMX_SIMD value -> (architecture, lscpu features it needs). Ordered best first per architecture.
SIMD_NEEDS = {
    "AVX_512": (X86, AVX512), "AVX2_256": (X86, {"avx2", "fma"}),
    "AVX2_128": (X86, {"avx2", "fma"}), "AVX_256": (X86, {"avx"}),
    "AVX_128_FMA": (X86, {"avx", "fma4"}), "SSE4.1": (X86, {"sse4_1"}), "SSE2": (X86, {"sse2"}),
    "ARM_SVE": (ARM, {"sve"}), "ARM_NEON_ASIMD": (ARM, {"asimd"}),
    "None": (X86 | ARM, set()),
}
# Compiler values that only make sense on one architecture: -mcpu (ARM), -march/-mprefer (x86)
def value_architectures(value):
    if value.startswith("-mcpu="):
        return ARM
    if value.startswith(("-march=", "-mprefer-vector-width=")):
        return X86
    return X86 | ARM


# x86-64 levels need these lscpu features
X86_LEVELS = {"-march=x86-64-v2": {"ssse3", "sse4_1", "sse4_2", "popcnt", "cx16"},
              "-march=x86-64-v3": {"avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "movbe"},
              "-march=x86-64-v4": AVX512}


def find_cuda_compiler():
    """Find nvcc on PATH or at the location specified by CUDACXX."""
    try:
        executable = shlex.split(os.environ.get("CUDACXX", "nvcc"))[0]
    except (IndexError, ValueError):
        return None
    return shutil.which(executable)


def profile_prerequisite_error(device, cpu_perf=True):
    """Return the first reason a complete host-side profile cannot run.

    cpu_perf=False skips the perf checks, for hosts where perf counters are
    blocked for users (perf_event_paranoid > 2), e.g. the student cluster GB10 nodes.
    """
    required = [*(["perf"] if cpu_perf else []),
                *(["hpcrun", "hpcstruct", "hpcprof"] if device == "cuda" else [])]
    missing = [tool for tool in required if not shutil.which(tool)]
    if missing:
        return "Profiling requires host tools: " + ", ".join(missing)
    if cpu_perf:
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
            with tempfile.TemporaryDirectory(prefix="gromacs-hpctoolkit-check-") as temporary:
                gpu_probe = subprocess.run(
                    ["hpcrun", "-o", str(Path(temporary) / "measurements"), "-e", "gpu=cuda",
                     shutil.which("true") or "true"], capture_output=True, text=True,
                )
        except OSError as error:
            return f"Cannot run HPCToolkit CUDA probe: {error}"
        if gpu_probe.returncode:
            detail = (gpu_probe.stderr or gpu_probe.stdout).strip()
            return f"HPCToolkit CUDA probe failed: {detail or f'exit {gpu_probe.returncode}'}"
    return None


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


def compile_check(executable: str, flag: str, *, link: bool = False) -> bool:
    """Can this compiler take the flag? Links too for OpenMP/LTO-like flags."""
    if not shutil.which(executable):
        return False
    language = "c++" if executable.endswith("++") else "c"
    command = [executable, "-x", language, "-", flag, "-o", "/dev/null"]
    if not link:
        command.insert(1, "-c")
    result = subprocess.run(command, input="int main(void) { return 0; }", text=True,
                            capture_output=True, check=False)
    return result.returncode == 0


def toolchain_supports(flag: str, *, link: bool = False) -> bool:
    return all(compile_check(exe, flag, link=link) for exe in COMPILERS)


def simd_levels(architecture, features):
    """GMX_SIMD values this CPU can run, best first."""
    return [level for level, (arches, needs) in SIMD_NEEDS.items()
            if architecture in arches and needs <= features]


def filter_catalog(catalog: dict, architecture: str, features: set[str], device: str) -> dict:
    """Intersect catalog choices with the detected CPU and installed compilers.

    The baseline (first value) stays first. GMX_SIMD gets the best level of this
    CPU as baseline, like a default GROMACS build (GMX_SIMD=AUTO) would choose.
    """
    if not all(shutil.which(exe) for exe in COMPILERS):
        raise ValueError("clang and clang++ must be installed")
    cpu = {}
    for name, values in catalog["cpu"].items():
        if name == "GMX_SIMD":
            supported = simd_levels(architecture, features)
            values = [level for level in supported if level in values]
        elif name == "GMX_SIMD_ARM_SVE_LENGTH" and "sve" not in features:
            continue
        elif name == "GMX_DOUBLE" and device == "cuda":
            values = ["OFF"]   # GPU builds are mixed precision only
        elif name == "CMAKE_INTERPROCEDURAL_OPTIMIZATION" and not toolchain_supports("-flto", link=True):
            values = ["OFF"]
        if values:
            cpu[name] = values
    compiler = {}
    for name, values in catalog["compiler"].items():
        eligible = [value for value in values
                    if architecture in value_architectures(value)
                    and X86_LEVELS.get(value, set()) <= features
                    and toolchain_supports(value, link=value.startswith("-flto"))]
        # the baseline must survive, otherwise the group is dropped (never silently re-based)
        if eligible and eligible[0] == values[0]:
            compiler[name] = eligible
    device_flags = dict(catalog["device"].get(device.upper(), {})) if device != "cpu" else {}
    device_flags.pop("GMX_GPU", None)   # set from --device, not sampled
    # nvcc flags (CMAKE_CUDA_FLAGS) only exist for CUDA builds
    cuda_compiler = catalog.get("cuda_compiler", {}) if device == "cuda" else {}
    known = cpu.keys() | compiler.keys() | device_flags.keys() | cuda_compiler.keys() | {"GMX_GPU"}
    rules = [rule for rule in catalog.get("invalid_combinations", []) if rule.keys() <= known]
    return {"cpu": cpu, "device": device_flags, "compiler": compiler,
            "cuda_compiler": cuda_compiler, "invalid_combinations": rules}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "flags.json")
    parser.add_argument("--output", type=Path, default=ROOT / "gromacs-performance-flags-system.json")
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
