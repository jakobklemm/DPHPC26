"""
What is specific to GROMACS when reading its runs (see load.py for the rest).

Runs come from the XaaS gromacs harness (gromacs_xaas_<date>.zip):
    build-<run id>.json  options    -> build configuration
    benchmark.json       mean       -> performance (ns/day simulated, higher is better)

`options` holds the CMake options (GMX_SIMD, GMX_DOUBLE, ...) and also the
compiler settings the harness turns into flags: compiler, opt_level, native, lto.
"""

APPLICATION = "gromacs"  # project_name in source.yml

# Options whose value follows from the hardware, {option: rule(hardware) -> value or None if not passed}
HARDWARE_DETERMINED = {}

# instruction set a GMX_SIMD value needs, as hardware feature
SIMD_NEEDS = {"AVX_512": "avx512", "AVX2_256": "avx2", "AVX2_128": "avx2", "ARM_NEON_ASIMD": "neon", "ARM_SVE": "sve"}


# can this configuration be built and run on this hardware?
def valid(config, hardware):
    # GMX_SIMD is a real choice (None, or AVX2_256 on an AVX-512 machine are
    # valid builds), but only among the instruction sets the CPU has
    simd = config.get("GMX_SIMD")
    if simd in SIMD_NEEDS and not hardware[SIMD_NEEDS[simd]]:
        return False
    # GMX_BUILD_OWN_FFTW only matters for FFTW: the harness writes n/a for fftpack
    own_fftw = config.get("GMX_BUILD_OWN_FFTW")
    if own_fftw is not None and (own_fftw == "n/a") != (config.get("GMX_FFT_LIBRARY") == "fftpack"):
        return False
    return True

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
