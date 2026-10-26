"""
What is specific to GROMACS when reading its runs (see load.py for the rest).

Runs come from the XaaS gromacs harness (gromacs_xaas_<date>.zip):
    build-<run id>.json  options    -> build configuration
    benchmark.json       mean       -> performance (ns/day simulated, higher is better)

`options` holds the CMake options (GMX_SIMD, GMX_DOUBLE, ...) and also the
compiler settings the harness turns into flags: compiler, opt_level, native, lto.
"""

from .validation import (BOOL, COMMON_SCHEMA, enabled, isa_errors, cuda_errors,
                         require, schema_errors, toolchain_errors, value)

APPLICATION = "gromacs"  # project_name in source.yml

# Options whose value follows from the hardware, {option: rule(hardware) -> value or None if not passed}
HARDWARE_DETERMINED = {}

# Supported harness profile. Source for option domains and dependencies:
# https://manual.gromacs.org/documentation/current/install-guide/index.html
# https://github.com/gromacs/gromacs/blob/main/CMakeLists.txt
SIMD_NEEDS = {
    "SSE2": ("x86", ("sse2",)), "SSE4.1": ("x86", ("sse41",)),
    "AVX_128_FMA": ("x86", ("avx", "fma")), "AVX_256": ("x86", ("avx",)),
    "AVX2_128": ("x86", ("avx2", "fma")), "AVX2_256": ("x86", ("avx2", "fma")),
    "AVX_512": ("x86", ("avx512", "avx512bw", "avx512dq", "avx512vl")),
    "ARM_NEON_ASIMD": ("arm", ("neon",)), "ARM_SVE": ("arm", ("sve",)),
    "IBM_VSX": ("power", ("vsx",)),
}
BOOLEAN_OPTIONS = {"GMX_DOUBLE", "GMX_MPI", "GMX_THREAD_MPI", "GMX_OPENMP",
                   "GMX_USE_RDTSCP", "GMX_BUILD_MDRUN_ONLY", "GMX_PREFER_STATIC_LIBS",
                   "GMX_EXTERNAL_BLAS", "GMX_EXTERNAL_LAPACK", "GMX_USE_TNG", "GMX_HWLOC",
                   "GMX_FFTW_USE_WISDOM", "GMX_DISABLE_FFTW_MEASURE", "native", "lto"}
SCHEMA = {
    **COMMON_SCHEMA, **dict.fromkeys(BOOLEAN_OPTIONS, BOOL),
    "GMX_SIMD": set(SIMD_NEEDS) | {"AUTO", "None", "NONE", "Reference", "REFERENCE"},
    "GMX_GPU": {"OFF", "CUDA", "OpenCL", "OPENCL", "SYCL", "HIP"},
    "GMX_FFT_LIBRARY": {"fftw3", "FFTW3", "mkl", "MKL", "fftpack", "fftpack[built-in]", "FFTPACK"},
    "GMX_BUILD_OWN_FFTW": BOOL | {"n/a"},
    "compiler": {"gcc", "clang", "icx", "icc"},
    "opt_level": {"0", "1", "2", "3", "s", "z", "fast", "O0", "O1", "O2", "O3", "Os", "Oz", "Ofast",
                  "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Ofast"},
    "extra_flags": lambda text: "\x00" not in text,
}


# can this configuration be built and run on this hardware?
def validation_errors(config, hardware):
    errors = schema_errors(config, SCHEMA)
    if errors:
        return errors
    # SIMD choice → CPU architecture and instruction requirements
    simd = value(config, "GMX_SIMD")
    if simd in SIMD_NEEDS:
        arch, features = SIMD_NEEDS[simd]
        errors.extend(isa_errors(hardware, features, f"GMX_SIMD={simd}", arch))
    # FFTW ownership meaningful only for the FFTW backend
    fft = value(config, "GMX_FFT_LIBRARY", "fftw3").lower()
    own_fftw = value(config, "GMX_BUILD_OWN_FFTW")
    if own_fftw == "n/a" and fft == "fftw3":
        errors.append("dependencies: GMX_BUILD_OWN_FFTW=n/a requires a non-FFTW FFT library")
    if enabled(config, "GMX_BUILD_OWN_FFTW") and fft != "fftw3":
        errors.append("dependencies: GMX_BUILD_OWN_FFTW=ON requires GMX_FFT_LIBRARY=fftw3")
    # Explicit MPI/thread-MPI conflict; default thread-MPI disabled by MPI
    if enabled(config, "GMX_MPI") and enabled(config, "GMX_THREAD_MPI"):
        errors.append("dependencies: GMX_MPI and GMX_THREAD_MPI are mutually exclusive")
    gpu = value(config, "GMX_GPU", "OFF").upper()
    if gpu != "OFF" and enabled(config, "GMX_DOUBLE"):
        errors.append("dependencies: GROMACS GPU acceleration requires mixed precision (GMX_DOUBLE=OFF)")
    if gpu == "CUDA":
        errors.extend(cuda_errors(config, hardware, "GMX_GPU=CUDA"))
    elif gpu != "OFF":
        require(errors, hardware, gpu.lower(), f"toolchain: GMX_GPU={gpu} requires that backend")
    if fft == "fftw3" and not enabled(config, "GMX_BUILD_OWN_FFTW"):
        # External FFTW precision matching GROMACS precision
        precision = "fftw_double" if enabled(config, "GMX_DOUBLE") else "fftw_single"
        require(errors, hardware, precision, f"toolchain: external FFTW requires {precision}")
        require(errors, hardware, "fftw", "toolchain: external FFTW is unavailable")
    elif fft == "mkl":
        require(errors, hardware, "mkl", "toolchain: GMX_FFT_LIBRARY=mkl requires MKL")
    if enabled(config, "GMX_BUILD_OWN_FFTW") and value(config, "CMAKE_GENERATOR") == "Ninja":
        errors.append("toolchain: building bundled FFTW is unsupported with Ninja")
    for option, feature in {"GMX_MPI": "mpi", "GMX_EXTERNAL_BLAS": "blas",
                            "GMX_EXTERNAL_LAPACK": "lapack", "GMX_HWLOC": "hwloc"}.items():
        if enabled(config, option):
            require(errors, hardware, feature, f"toolchain: {option} requires {feature}")
    # Shared compiler, flag, OpenMP and LTO checks
    errors.extend(toolchain_errors(config, hardware, openmp=enabled(config, "GMX_OPENMP", True),
                                   lto=enabled(config, "lto")))
    return list(dict.fromkeys(errors))


def valid(config, hardware):
    return not validation_errors(config, hardware)

# options that are not CMake definitions but compiler settings of the harness
HARNESS_OPTIONS = {"compiler", "opt_level", "native", "lto", "extra_flags"}


# extract build config from build-<run>.json
def build_config(build):
    config = {}
    for option, value in build["options"].items():
        if isinstance(value, bool):
            value = "ON" if value else "OFF"
        # only some runs list extra_flags, and always empty
        if option == "extra_flags" and not value:
            continue
        config[option] = str(value)
    return config


# extract performance from benchmark.json: mean over the measured runs (warmup excluded)
def performance(benchmark):
    assert benchmark["higher_is_better"], f"expected a higher-is-better metric, got {benchmark['metric']}"
    return benchmark["mean"]


# how to pass one option to the build
def build_argument(option, value):
    return f"{option}={value}" if option in HARNESS_OPTIONS else f"-D{option}={value}"
