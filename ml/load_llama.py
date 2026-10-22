"""
What is specific to llama.cpp when reading its runs (see load.py for the rest).

Runs come from llama_gpu_bench.py:
    build-<run id>.json  cmake_definitions_in_order -> build configuration
    benchmark.json       llama-bench output         -> performance (tokens per second)
"""

APPLICATION = "llama.cpp"  # project_name in source.yml

# The default value CMake uses for an option when the build does not pass it.
# to detect that a build that does not pass GGML_LTO and a build that passes -DGGML_LTO=OFF are the same build
CMAKE_DEFAULTS = {"GGML_LTO": "OFF", "GGML_OPENMP": "ON", "GGML_CUDA_FA": "ON"}

# Options whose value follows from the hardware, {option: rule(hardware) -> value or None if not passed}
HARDWARE_DETERMINED = {
    "CMAKE_CUDA_ARCHITECTURES":
        lambda hw: str(round(hw["gpu_compute_capability"] * 10)) if hw["n_gpus"] else None,
}


# can this configuration be built and run on this hardware? (no restrictions known yet)
def valid(config, hardware):
    return True


# extract build config from build-<run>.json
def build_config(build):
    config = dict(CMAKE_DEFAULTS)
    for flag in build["cmake_definitions_in_order"]:
        name, _, value = flag[len("-D"):].partition("=")
        config[name] = value.strip('"')
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


# how to pass one option to the build: all llama.cpp options are CMake definitions
def build_argument(option, value):
    return f"-D{option}={value}"
