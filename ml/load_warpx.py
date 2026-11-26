"""
What is specific to WarpX when reading its runs (see load.py for the rest).

Runs come from warpx/warpx_bench-profile.py (via warpx_bench-random.py):
    build.json      cmake          -> CMake options (WarpX_*, AMReX_*, ...)
                    compiler_flags -> one option per flag group of flags.json, e.g. "-fvectorize"
    benchmark.json  avg_particle_steps_per_second -> performance (particle-steps/s, higher is better)

The workload is fixed (warpx/inputs_bench_3d, 20 steps, 2^21 particles). The thread count
follows from WarpX_COMPUTE: OMP builds run with the harness' --threads, NOACC builds serially.
"""

from .validation import (BOOL, COMMON_SCHEMA, compiler_flag_option, compiler_flags, isa_errors,
                         schema_errors, toolchain_errors, value)

APPLICATION = "warpx"  # project_name in source.yml
UNIT = "particle-steps/s"

# Options whose value follows from the hardware: none for the CPU builds.
HARDWARE_DETERMINED = {}

# Domains of flags.json "cpu" and the CPU entry of "device"
BOOLEAN_OPTIONS = {"WarpX_SIMD", "WarpX_FASTMATH", "WarpX_IPO", "WarpX_UNITY_BUILD", "WarpX_EB", "WarpX_QED",
                   "AMReX_FLATTEN_FOR", "AMReX_ARRAY4_64BIT_INDEX", "AMReX_MATH_ERRNO",
                   "ABLASTR_POSITION_INDEPENDENT_CODE"}
PRECISION = {"DOUBLE", "SINGLE"}
SCHEMA = {**COMMON_SCHEMA, **dict.fromkeys(BOOLEAN_OPTIONS, BOOL),
          "WarpX_COMPUTE": {"OMP", "NOACC"},
          "WarpX_PRECISION": PRECISION, "WarpX_PARTICLE_PRECISION": PRECISION}
PREFIXES = ("WarpX_", "AMReX_", "ABLASTR_")


def validation_errors(config, hardware):
    compiler = compiler_flags(APPLICATION)
    errors = schema_errors(config, {**SCHEMA, **compiler})
    if errors:
        return errors
    flags = [value(config, option) for option in compiler if value(config, option)]
    for flag in flags:
        if flag.startswith("-mprefer-vector-width"):
            errors.extend(isa_errors(hardware, (), flag))
    # -march levels need their instructions; the harness builds with clang and OpenMP for OMP
    errors.extend(toolchain_errors({**config, "extra_flags": " ".join(flags)}, hardware,
                                   openmp=value(config, "WarpX_COMPUTE") == "OMP",
                                   lto=value(config, "WarpX_IPO", "OFF").upper() in {"ON", "TRUE", "1", "YES"}))
    return list(dict.fromkeys(errors))


def valid(config, hardware):
    return not validation_errors(config, hardware)


# extract build config from build.json
def build_config(build):
    config = {option: str(v) for option, v in build["cmake"].items()}
    for flag in build.get("compiler_flags", []):
        config[compiler_flag_option(flag)] = flag
    return config


# extract performance from benchmark.json: one entry, mean over the timed repetitions
def performance(benchmark):
    assert len(benchmark) == 1, f"expected one benchmark entry, got {len(benchmark)}"
    return float(benchmark[0]["avg_particle_steps_per_second"])


# how to pass one option to the build
def build_argument(option, value):
    return value if option in compiler_flags(APPLICATION) else f"-D{option}={value}"
