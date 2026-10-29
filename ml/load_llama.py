"""
What is specific to llama.cpp when reading its runs (see load.py for the rest).

Runs come from llama_gpu_bench.py (build-<run id>.json) or llama_gpu_bench-profile.py (build.json):
    cmake_definitions_in_order -> CMake options
    configure_command          -> compiler and build type (build.json only)
    compiler_flags             -> one option per flag group of flags.json, e.g. "-fvectorize" (build.json only)
    benchmark.json             llama-bench output -> performance (tokens per second)
"""

import json
from pathlib import Path

from .validation import (BOOL, COMMON_SCHEMA, cuda_errors, enabled,
                         isa_errors, require, schema_errors, toolchain_errors, value)

APPLICATION = "llama.cpp"  # project_name in source.yml

# Folders in llama/results/ not read by default: the threadripper runs come from the old
# harness, which passes only the options it changes and records no compiler flags.
# Explicitly given paths (--data) are still read.
IGNORED_RESULTS = {"threadripper_run0"}

# The default value CMake uses for an option when the build does not pass it.
# to detect that a build that does not pass GGML_LTO and a build that passes -DGGML_LTO=OFF are the same build
CMAKE_DEFAULTS = {"GGML_LTO": "OFF", "GGML_OPENMP": "ON", "GGML_CUDA_FA": "ON"}

# Options whose value follows from the hardware, {option: rule(hardware) -> value or None if not passed}
HARDWARE_DETERMINED = {
    "CMAKE_CUDA_ARCHITECTURES":
        lambda hw: str(round(hw["gpu_compute_capability"] * 10))
        if hw.get("n_gpus", 0) > 0 and hw.get("gpu_compute_capability", 0) > 0 else None,
}


# Supported prediction profile: the repository's flags.json plus the build
# arguments emitted by the harness. Extend this schema when adding new options.
# Sources: https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt
# https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/CMakeLists.txt
CPU_FEATURES = {
    "GGML_SSE42": ("sse42",), "GGML_AVX": ("avx",), "GGML_AVX2": ("avx2",),
    "GGML_FMA": ("fma",), "GGML_F16C": ("f16c",), "GGML_BMI2": ("bmi2",),
    "GGML_AVX_VNNI": ("avx_vnni",),
    "GGML_AVX512": ("avx512", "avx512bw", "avx512dq", "avx512vl", "avx512cd"),
    "GGML_AVX512_VBMI": ("avx512", "avx512_vbmi"),
    "GGML_AVX512_VNNI": ("avx512", "avx512_vnni"),
    "GGML_AVX512_BF16": ("avx512", "avx512_bf16"),
}
BOOLEAN_OPTIONS = set(CPU_FEATURES) | {
    "GGML_CUDA", "GGML_NATIVE", "GGML_LTO", "GGML_CPU", "GGML_CPU_REPACK",
    "GGML_LLAMAFILE", "GGML_OPENMP", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
    "GGML_CUDA_NO_VMM", "GGML_CUDA_FA", "GGML_CUDA_FA_ALL_QUANTS", "GGML_CUDA_GRAPHS",
    "GGML_CUDA_NO_PEER_COPY", "GGML_CUDA_F16", "GGML_HIP", "GGML_SYCL", "GGML_VULKAN",
    "GGML_METAL", "GGML_BLAS", "GGML_BACKEND_DL", "LLAMA_CURL",
}
# flag group of a compiler flag: "-fno-vectorize" -> "-fvectorize", "-O2" -> "-O", "-march=native" -> "-march"
def compiler_flag_option(flag):
    if flag.startswith("-O"):
        return "-O"
    name = flag.partition("=")[0]
    return "-f" + name.removeprefix("-fno-") if name.startswith("-fno-") else name


# Compiler flags of the harness: the flag groups of flags.json "compiler", {option: check(flag)}.
# The option is the flag group, its value the flag passed, e.g. "-fvectorize": "-fno-vectorize".
# Any flag of the group is accepted, also the harness' "off" values that flags.json does not list.
FLAGS_FILE = Path(__file__).resolve().parent.parent / "llama" / "flags.json"
COMPILER_FLAGS = {group: lambda flag, group=group: compiler_flag_option(flag) == group
                  for group in json.loads(FLAGS_FILE.read_text())["compiler"]}
# Toolchain choices read from configure_command; the tuned options come from the other fields
TOOLCHAIN = {"CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"}
SCHEMA = {**COMMON_SCHEMA, **dict.fromkeys(BOOLEAN_OPTIONS, BOOL), **COMPILER_FLAGS,
          "GGML_BLAS_VENDOR": {"Generic", "OpenBLAS", "FLAME", "Intel10_64lp", "Apple", "All"}}


def validation_errors(config, hardware):
    errors = schema_errors(config, SCHEMA)
    if errors:
        return errors
    # CUDA tuning constraints only with an active CUDA backend
    if enabled(config, "GGML_CUDA"):
        errors.extend(cuda_errors(config, hardware, "GGML_CUDA"))
        if enabled(config, "GGML_CUDA_FORCE_MMQ") and enabled(config, "GGML_CUDA_FORCE_CUBLAS"):
            errors.append("dependencies: CUDA FORCE_MMQ and FORCE_CUBLAS are mutually exclusive")
    if enabled(config, "GGML_BACKEND_DL") and not enabled(config, "BUILD_SHARED_LIBS", True):
        errors.append("dependencies: GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
    # Native CPU tuning bypassing individual x86 instruction switches
    arch = str(hardware.get("cpu_arch", "unknown")).lower()
    x86 = arch in {"x86_64", "amd64", "x86", "i686"}
    if (x86 or arch == "unknown") and enabled(config, "GGML_CPU", True) and not enabled(config, "GGML_NATIVE", True):
        # Implicit x86 instruction defaults with native tuning disabled
        default_isa = {"GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"}
        for option, features in CPU_FEATURES.items():
            if enabled(config, option, x86 and option in default_isa):
                errors.extend(isa_errors(hardware, features, option))
    for option, feature in {"GGML_HIP": "hip", "GGML_SYCL": "sycl", "GGML_VULKAN": "vulkan",
                            "GGML_METAL": "metal"}.items():
        if enabled(config, option):
            require(errors, hardware, feature, f"toolchain: {option} requires {feature} support")
    if enabled(config, "GGML_BLAS"):
        # Selected BLAS vendor → required library
        vendor = value(config, "GGML_BLAS_VENDOR", "Generic")
        feature = {"OpenBLAS": "openblas", "Intel10_64lp": "mkl", "Apple": "accelerate"}.get(vendor, "blas")
        require(errors, hardware, feature, f"toolchain: GGML_BLAS requires {vendor} BLAS")
    if enabled(config, "LLAMA_CURL"):
        require(errors, hardware, "curl", "toolchain: LLAMA_CURL requires libcurl")
    # Compiler flags checked like extra flags of the build (e.g. -march=x86-64-v3 needs AVX2)
    flags = [value(config, option) for option in COMPILER_FLAGS if value(config, option)]
    for flag in flags:
        if flag.startswith("-mprefer-vector-width"):
            errors.extend(isa_errors(hardware, (), flag))
    errors.extend(toolchain_errors({**config, "extra_flags": " ".join(flags)}, hardware,
                                   openmp=enabled(config, "GGML_OPENMP", True),
                                   lto=enabled(config, "GGML_LTO")))
    return list(dict.fromkeys(errors))


def valid(config, hardware):
    return not validation_errors(config, hardware)


# extract build config from build-<run>.json or build.json
def build_config(build):
    config = dict(CMAKE_DEFAULTS)
    for flag in build["cmake_definitions_in_order"]:
        name, _, value = flag[len("-D"):].partition("=")
        config[name] = value.strip('"')
    for argument in build.get("configure_command", []):
        name, _, value = argument.removeprefix("-D").partition("=")
        if argument.startswith("-D") and name in TOOLCHAIN:
            config[name] = value
    for flag in build.get("compiler_flags", []):
        config[compiler_flag_option(flag)] = flag
    return config


# extract performance metrics from benchmark.json
def performance(benchmark):
    """
    benchmark.json holds one entry per test, currently for llama two:
      - prompt processing (n_prompt=512): reads a prompt, compute-bound
      - token generation  (n_gen=1024):   writes an answer, memory-bound
    """
    # The tests are combined into the total time for the whole benchmark
    # workload (here: read 512 tokens, then generate 1024). Each test thus
    # counts as much as the time it takes: generation takes ~10.5 s and
    # prompt processing ~1.3 s, so generation weighs about 8 times more.
    total_seconds = sum(test["avg_ns"] for test in benchmark) / 1e9
    total_tokens = sum(test["n_prompt"] + test["n_gen"] for test in benchmark)
    # Returned as tokens per second over the whole workload, so that higher is
    # better like for the other applications.
    return total_tokens / total_seconds


# how to pass one option to the build: compiler flags as they are, the rest as CMake definitions
def build_argument(option, value):
    return value if option in COMPILER_FLAGS else f"-D{option}={value}"
