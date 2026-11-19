# Local llama.cpp tuning

Build and benchmark **directly on the host**, with clang/clang++ and CMake. No
XaaS or container runtime is used. By default the scripts look for
`llama/llama.cpp/`, `llama/model.gguf`, and `llama/flags.json` **beside the
scripts**, regardless of the directory you run them from. Pass `--source`,
`--model`, and/or `--flags-file` to use other paths. Cloning/downloading the
source and model once is enough for subsequent runs.

Prerequisites: Python 3, CMake, clang, clang++, `lscpu`, and llama.cpp's normal
build dependencies. CUDA runs additionally need the CUDA toolkit and an NVIDIA
GPU; Vulkan runs need the Vulkan development packages and a Vulkan-capable GPU.
CPU runs need neither GPU toolkit. CUDA compilation uses `nvcc` from the host
toolkit and clang++ as its host compiler. A working NVIDIA driver or Docker
CUDA image does not by itself put `nvcc` on the host. For the default source location, use
`git clone https://github.com/ggml-org/llama.cpp.git llama/llama.cpp` from the
repository root; place a GGUF model at `llama/model.gguf`, or specify its path.

```sh
# With the default script-local source, model and flag catalog:
python3 llama/llama_gpu_bench-random.py random --device cpu --count 3 \
  --prompt 16 --generate 16 --repetitions 1

# One configuration for each selectable flag, with every other flag OFF:
python3 llama/llama_gpu_bench-random.py sweep \
  --source /path/to/llama.cpp --model /path/to/model.gguf --device cpu \
  --workers 4 --prompt 32 --generate 32 --repetitions 1

python3 llama/llama_gpu_bench-random.py random \
  --source /path/to/llama.cpp --model /path/to/model.gguf --device cpu \
  --count 50

python3 llama/llama_gpu_bench-random.py random \
  --source /path/to/llama.cpp --model /path/to/model.gguf \
  --device cuda --cuda-arch 86 --count 5
```

`--cuda-arch` is mandatory with CUDA and forbidden with other devices. The
backend choice is fixed per invocation, not randomly toggled. 

Vulkan configurations can be inspected with `--dry-run`, but profiled execution
is rejected until a Vulkan GPU metrics collector is available; HPCToolkit's
`gpu=cuda` event source cannot profile Vulkan. By default, CPU-only runs require
timing and CPU profiling; CUDA runs require all three measurement stages.
Use `--no-profile` for timing-only runs, including Vulkan. Use `--dry-run` to
preview configurations without building or saving a plan. The root
`llama/flags.json` is the canonical flag catalog. Supply a different catalog
with `--flags-file /path/to/flags.json`; the same path is passed to each build
for invalid-combination checks. A host-filtered catalog can be saved with
`--flags-output /path/to/eligible.json`. Relative paths supplied on the command
line are resolved against the shell's current directory.

Every configuration specifies each eligible CMake, clang, and Vulkan runtime
flag explicitly. Unsupported CPU flags are explicitly OFF. The optimization
level defaults to `-O3` for Release builds; `-O0` is not sampled because it
produces an unoptimized benchmark. Vulkan's
`GGML_VK_DISABLE_*` flags are runtime environment settings (`0` or `1`), not
CMake definitions. Invalid combinations are skipped before a build: for
example, CUDA's mutually exclusive FORCE_MMQ/FORCE_CUBLAS options and
FA_ALL_QUANTS without FA. Fast-math and Ofast cannot be used with ggml's
non-finite arithmetic, so the host filter excludes them instead of silently
negating them with another compiler switch.

The default build directory is `llama/build/`, reused across configurations;
CMake reconfigures and incrementally builds only as needed. The flag filter,
host snapshot, and model hash are calculated once per invocation. Results are
in `llama/bench-results/` by default: `sweep-plan.json` contains **all** planned
sweep configurations. Random mode samples at each step without writing a plan.
`<mode>-executions.json` is the summary of executed configurations, with every
flag's value and each execution's result, including failures. Each run directory
has `build.json`, `benchmark.json`, logs, and system metadata. With profiling
enabled, a completed CPU run also has `cpu-metrics.json`; a completed CUDA run
additionally has `gpu-metrics.json`.
`build.json` records each stage's status. Re-running the same mode and results directory
resumes the run; random mode replays the seeded sampler to skip completed entries
and retry failures with the same flags. Increase `--count` to extend a random run.
`<mode>-settings.json` prevents accidentally resuming with a different model,
benchmark workload, or sampling method/eligible flags. Old independent-toggle
and fixed-size `subset` runs are not resumed by this sampler. Use a new results directory for
different settings. `--build-dir` and `--results-dir` can be set independently.
Failed executions are recorded and reported with a nonzero script exit code.
After fixing the environment, rerun with the same arguments: completed entries
are skipped, failed entries are retried, and prior attempts remain listed in
`previous_attempts`.

```sh
python3 llama/llama_gpu_bench-random.py random \
  --source /path/to/llama.cpp --model /path/to/model.gguf \
  --device cuda --cuda-arch 75 --count 1 \
  --results-dir /path/to/bench-results-profiled
```
