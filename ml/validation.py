"""Static build constraints shared by the two application profiles.

Missing discovery data is not evidence of incompatibility. These checks never
consult training measurements or run compilers on the prediction host.
"""

import math
import json
import os
import re
import shlex
from pathlib import Path, PurePath
from functools import lru_cache


TRUE = {"ON", "YES", "TRUE", "1"}
FALSE = {"OFF", "NO", "FALSE", "0"}
BOOL = TRUE | FALSE


@lru_cache(maxsize=None)
def flag_catalog(application):
    directory = {"llama.cpp": "llama", "gromacs": "gromacs"}[application]
    root = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent))
    return json.loads((root / directory / "flags.json").read_text())


def catalog_errors(config, application, defaults=None):
    """Honor the same experiment exclusions as the benchmark flag catalog.

    These include sampling policy, not only configurations that cannot compile.
    Existing historical exclusions are consolidated into the application
    catalog; prediction does not depend on result folders being available.
    """
    effective = {**(defaults or {}), **config}

    def matches(option, expected):
        actual = value(effective, option)
        if expected in BOOL:
            return actual is not None and actual.upper() in (TRUE if expected in TRUE else FALSE)
        return actual == expected

    return ["catalog: excluded combination " + ", ".join(f"{k}={v}" for k, v in rule.items())
            for rule in flag_catalog(application).get("invalid_combinations", [])
            if all(matches(option, expected) for option, expected in rule.items())]


def compiler_flag_option(flag):
    if flag.startswith("-O"):
        return "-O"
    name = flag.partition("=")[0]
    return "-f" + name.removeprefix("-fno-") if name.startswith("-fno-") else name


def value(config, option, default=None):
    # Omitted arguments → application defaults
    result = config.get(option)
    return default if result is None or result == "default" else str(result)


def enabled(config, option, default=False):
    return str(value(config, option, "ON" if default else "OFF")).upper() in TRUE


def known(hardware, feature):
    """Return True/False for known capabilities, None for absent information."""
    result = hardware.get(feature)
    if result is None or (isinstance(result, float) and math.isnan(result)):
        return None
    text = str(result).upper()
    return True if text in TRUE else False if text in FALSE else None


def require(errors, hardware, feature, reason):
    # Rejection only for explicitly unavailable features
    if known(hardware, feature) is False:
        errors.append(reason)


def nonempty(text):
    return isinstance(text, str) and bool(text.strip()) and "\x00" not in text


def positive_integer(text):
    return bool(re.fullmatch(r"[1-9][0-9]*", text))


def cuda_arch_list(text):
    return text in {"native", "all", "all-major"} or all(
        re.fullmatch(r"[1-9][0-9]+(?:-real|-virtual)?", arch)
        for arch in text.split(";")
    )


# Shared CMake option domains and value validators
COMMON_SCHEMA = {
    "CMAKE_BUILD_TYPE": {"Debug", "Release", "RelWithDebInfo", "MinSizeRel", ""},
    "CMAKE_C_COMPILER": nonempty,
    "CMAKE_CXX_COMPILER": nonempty,
    "CMAKE_CUDA_COMPILER": nonempty,
    "CMAKE_CUDA_HOST_COMPILER": nonempty,
    "CMAKE_CUDA_ARCHITECTURES": cuda_arch_list,
    "CMAKE_CXX_STANDARD": {"17", "20", "23", "26"},
    "CMAKE_INTERPROCEDURAL_OPTIMIZATION": BOOL,
    "BUILD_SHARED_LIBS": BOOL,
    "CMAKE_GENERATOR": nonempty,
    "CMAKE_PREFIX_PATH": lambda text: "\x00" not in text,
    "CMAKE_INSTALL_PREFIX": nonempty,
}
for language in ("C", "CXX", "CUDA"):
    for suffix in ("", "_DEBUG", "_RELEASE", "_RELWITHDEBINFO", "_MINSIZEREL"):
        COMMON_SCHEMA[f"CMAKE_{language}_FLAGS{suffix}"] = lambda text: "\x00" not in text


def schema_errors(config, schema):
    errors = []
    for option, raw in config.items():
        # 'default' → argument omitted by describe()
        if raw == "default":
            continue
        check = schema.get(option)
        if check is None:
            errors.append(f"schema: unsupported option {option}")
        elif not isinstance(raw, (str, bool, int, float)):
            errors.append(f"schema: invalid value for {option}: {raw!r}")
        else:
            text = str(raw)
            if not callable(check) and BOOL <= check:
                # Case-insensitive CMake boolean spellings
                accepted = text.upper() in BOOL or text in check
            else:
                accepted = check(text) if callable(check) else text in check
            if not accepted:
                errors.append(f"schema: invalid value for {option}: {raw!r}")
    return errors


COMPILER_PAIRS = {"gcc": ("gcc", "g++"), "clang": ("clang", "clang++"),
                  "icx": ("icx", "icpx"), "icc": ("icc", "icpc")}


def compiler_name(command):
    name = PurePath(command).name
    # Version suffix removal, e.g. gcc-14 → gcc
    return re.sub(r"-\d+(?:\.\d+)*$", "", name)


def compiler_family(command):
    name = compiler_name(command)
    return next((family for family, pair in COMPILER_PAIRS.items() if name in pair), None)


def isa_errors(hardware, features, label, architecture="x86"):
    arch = str(hardware.get("cpu_arch", "unknown")).lower()
    families = {"x86": {"x86_64", "amd64", "x86", "i686"},
                "arm": {"aarch64", "arm64", "armv8", "armv8l"},
                "power": {"ppc64", "ppc64le", "powerpc64", "powerpc64le"}}
    if arch not in {"unknown", ""} and arch not in families[architecture]:
        return [f"hardware: {label} requires {architecture} hardware"]
    return [f"hardware: {label} requires {feature}" for feature in features
            if known(hardware, feature) is False]


def toolchain_errors(config, hardware, *, openmp=False, lto=False):
    errors = []
    selected = {}
    family = value(config, "compiler")
    # Harness compiler family → C/C++ executable pair
    if family in COMPILER_PAIRS:
        selected.update(zip(("C", "CXX"), COMPILER_PAIRS[family]))
    for language in ("C", "CXX"):
        # Explicit CMake compiler overriding the harness choice
        command = value(config, f"CMAKE_{language}_COMPILER", selected.get(language))
        if command:
            selected[language] = command
            require(errors, hardware, f"compiler_{language.lower()}_{compiler_name(command)}",
                    f"toolchain: {language} compiler {command} is unavailable")
    for language in ("C", "CXX"):
        command = selected.get(language, "default")
        family = compiler_family(command)
        if family:
            if openmp:
                require(errors, hardware, f"{family}_openmp", f"toolchain: {family} lacks OpenMP support")
            if lto or enabled(config, "CMAKE_INTERPROCEDURAL_OPTIMIZATION"):
                require(errors, hardware, f"{family}_lto", f"toolchain: {family} lacks LTO support")
        # Global flags + active build-type flags; parsing only
        build_type = value(config, "CMAKE_BUILD_TYPE", "").upper()
        names = [f"CMAKE_{language}_FLAGS"]
        if build_type:
            names.append(f"CMAKE_{language}_FLAGS_{build_type}")
        if value(config, "extra_flags") is not None:
            names.append("extra_flags")
        for option in names:
            try:
                flags = shlex.split(value(config, option, ""))
            except ValueError:
                errors.append(f"schema: malformed flags in {option}")
                continue
            for flag in flags:
                # Clang-compatible tree-vectorize aliases excluded from rejection
                # https://clang.llvm.org/docs/ClangCommandLineReference.html
                if family in {"clang", "icx"} and (flag in {"-ftree-loop-vectorize", "-fno-tree-loop-vectorize"}
                                                    or flag.startswith("-fvect-cost-model=")):
                    errors.append(f"toolchain: {flag} is a GCC-specific flag, selected {command}")
                if language == "CXX" and re.fullmatch(r"-std=(?:gnu|c)\+\+(?:98|03|11|14)", flag):
                    errors.append(f"toolchain: {flag} does not provide required C++17 support")
                isa = {"-mavx": ("avx",), "-mavx2": ("avx2",), "-mfma": ("fma",),
                       "-mf16c": ("f16c",), "-mbmi2": ("bmi2",), "-mavx512f": ("avx512",)}
                if flag in isa:
                    errors.extend(isa_errors(hardware, isa[flag], flag))
                if flag.startswith("-march=x86-64"):
                    # Cumulative instruction requirements for each x86-64 level
                    v2 = ("sse2", "sse41", "sse42", "ssse3", "popcnt", "cx16", "lahf_lm")
                    v3 = v2 + ("avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "movbe", "xsave", "lzcnt")
                    features = {"-march=x86-64": (), "-march=x86-64-v2": v2, "-march=x86-64-v3": v3,
                                "-march=x86-64-v4": v3 + ("avx512", "avx512bw", "avx512dq", "avx512vl", "avx512cd")}
                    errors.extend(isa_errors(hardware, features.get(flag, ()), flag))
    return list(dict.fromkeys(errors))


def discovery_features(system):
    """Scalar validation inputs that also survive dataframe/model round trips.

    XaaS vectorizations are NOT a full CPUID list (they omit FMA/BMI2).
    Only a supplied full 'Flags' list establishes absence of those features.
    Library discovery lists shared runtime libraries, not development packages;
    an empty inventory therefore does not establish that linking is impossible.
    """
    result = {}

    def state(flag):
        flag = known({"value": flag}, "value")
        return "unknown" if flag is None else "yes" if flag else "no"

    cpu = system.get("CPU Info", {})
    vectors = cpu.get("Supported Vectorizations")
    full = cpu.get("Flags")
    if isinstance(full, str):
        full = full.split()
    flags = {str(f).lower().replace(".", "_") for f in (full if full is not None else vectors or [])}
    # Discovery spelling variants → common feature names
    aliases = {"sse2": {"sse2"}, "sse41": {"sse4_1"}, "sse42": {"sse4_2"},
               "avx": {"avx"}, "fma": {"fma"}, "f16c": {"f16c"}, "bmi2": {"bmi2"},
               "avx_vnni": {"avx_vnni", "avxvnni"}, "avx512_vbmi": {"avx512vbmi", "avx512_vbmi"},
               "avx512_vnni": {"avx512_vnni", "avx512vnni"}, "avx512_bf16": {"avx512_bf16"},
               **{f"avx512{s}": {f"avx512{s}"} for s in ("bw", "dq", "vl", "cd")}, "vsx": {"vsx"},
               **{f: {f} for f in ("ssse3", "popcnt", "cx16", "lahf_lm", "movbe", "xsave")},
               "bmi1": {"bmi1", "bmi"}, "lzcnt": {"lzcnt", "abm"}}
    for feature, names in aliases.items():
        # Partial vectorization lists: no inferred FMA/BMI2 absence
        complete = full is not None or (vectors is not None and feature in {"avx", "sse2", "sse41", "sse42"})
        result[feature] = state(True if flags & names else False if complete else None)
    compilers = system.get("Compilers", {})
    for language in ("c", "cxx"):
        inventory = compilers.get(language, {})
        for name in {p[language == "cxx"] for p in COMPILER_PAIRS.values()}:
            description = inventory.get(name)
            present = None if description is None else str(description).lower() not in {"not found", "none", "", "unknown"}
            result[f"compiler_{language}_{name}"] = state(present)
    for backend in ("OpenCL", "Vulkan", "Metal"):
        record = system.get("GPU Backends", {}).get(backend)
        result[backend.lower()] = state(bool(record.get("version")) if record is not None else None)
    # Positive library evidence; incomplete inventories kept permissive
    libraries = str({k: system.get(k, {}) for k in
                     ("FFT Libraries", "linear algebra Libraries", "Parallel Libraries")}).lower()
    for feature, token in {"fftw": "fftw", "mkl": "mkl", "blas": "blas", "openblas": "openblas",
                           "lapack": "lapack", "mpi": "mpi"}.items():
        found = token in libraries
        if feature == "mpi" and "no mpi found" in libraries:
            result[feature] = "no"
        else:
            result[feature] = "yes" if found else "unknown"
    # Explicit build capabilities overriding discovery, e.g. gcc_lto=false
    capabilities = system.get("Build Capabilities", {})
    for feature in ("fftw_single", "fftw_double", "curl", "hwloc", "accelerate", "fftw", "mkl",
                    "blas", "openblas", "lapack", "mpi"):
        result[feature] = state(capabilities.get(feature)) if feature in capabilities else result.get(feature, "unknown")
    for family in COMPILER_PAIRS:
        for capability in ("openmp", "lto"):
            key = f"{family}_{capability}"
            result[key] = state(capabilities.get(key))
    return result


def cuda_errors(config, hardware, option):
    """Check runtime hardware and discovered backend, independently of the host."""
    errors = []
    require(errors, hardware, "cuda", f"toolchain: {option} requires a CUDA backend")
    if hardware.get("n_gpus") == 0:
        errors.append(f"hardware: {option} requires an NVIDIA GPU")
    archs = value(config, "CMAKE_CUDA_ARCHITECTURES")
    capability = hardware.get("gpu_compute_capability", 0)
    if archs and cuda_arch_list(archs) and archs not in {"native", "all", "all-major"} and capability:
        cc = round(float(capability) * 10)
        # Unsuffixed: native code + forward-compatible PTX
        # Real-only: same major SM version required
        if not any(int(a.split("-")[0]) <= cc and
                   (not a.endswith("-real") or int(a.split("-")[0]) // 10 == cc // 10)
                   for a in archs.split(";")):
            errors.append(f"hardware: CUDA targets {archs} cannot run on compute capability {capability}")
    return errors
