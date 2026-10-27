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
python3 llama/llama_gpu_bench-random.py subset --device cpu --count 3 \
  --prompt 16 --generate 16 --repetitions 1

# One configuration for each selectable flag, with every other flag OFF:
python3 llama/llama_gpu_bench-random.py sweep \
  --source /path/to/llama.cpp --model /path/to/model.gguf --device cpu \
  --workers 4 --prompt 32 --generate 32 --repetitions 1

# 50 distinct random configurations; exactly 7 flags ON in each by default:
python3 llama/llama_gpu_bench-random.py subset \
  --source /path/to/llama.cpp --model /path/to/model.gguf --device cpu \
  --count 50 --flags-per-build 7

# CUDA example:
python3 llama/llama_gpu_bench-random.py subset \
  --source /path/to/llama.cpp --model /path/to/model.gguf \
  --device cuda --cuda-arch 86 --count 5
```

`--cuda-arch` is mandatory with CUDA and forbidden with other devices. The
backend choice is fixed per invocation, not counted toward the N tuning flags.
Vulkan configurations can be inspected with `--dry-run`, but execution is
rejected until a Vulkan GPU metrics collector is available; HPCToolkit's
`gpu=cuda` event source cannot profile Vulkan. CPU-only runs require timing
and CPU profiling; CUDA runs require all three measurement stages.
Use `--dry-run` to print all configurations before building. The root
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
in `llama/bench-results/` by default: `<mode>-plan.json` contains **all**
planned configurations and `<mode>-executions.json` records their results,
including failures. Each run directory has `build.json`, `benchmark.json`,
logs, and system metadata. A completed CPU run has `benchmark.json` and
`cpu-metrics.json`; a completed CUDA run also has `gpu-metrics.json`.
`build.json` records each stage's status. Re-running the same mode and results directory
resumes the recorded plan; `<mode>-settings.json` prevents accidentally resuming
with a different model or benchmark workload. Use a new results directory for
different settings. `--build-dir` and `--results-dir` can be set independently.
Failed executions are recorded and reported with a nonzero script exit code.
After fixing the environment, rerun with the same arguments: completed entries
are skipped, failed entries are retried, and prior attempts remain listed in
`previous_attempts`.

### CUDA on Rocky Linux 10

Check `nvidia-smi`, `clang++ --version`, and `command -v nvcc` on the host.
If `nvcc` is already installed but is outside `PATH`, set, for example,
`export CUDACXX=/usr/local/cuda/bin/nvcc`. Check with `"$CUDACXX" --version`.
Otherwise install the CUDA **development toolkit** on the host (the driver
supplied to Docker containers is insufficient). NVIDIA's Rocky 10 packages
are in the `rhel10/x86_64` repository:

```sh
sudo dnf install clang cmake make python3 git libstdc++-devel dnf-plugins-core
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
sudo dnf install cuda-toolkit
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
nvcc --list-gpu-arch
```

Use a toolkit version compatible with your installed NVIDIA driver and host
Clang. The CUDA 12.8 guide does not list Rocky 10 as supported; NVIDIA's
current Linux installation guide does. For CUDA architecture 75, check that
`nvcc --list-gpu-arch` includes `compute_75`. Then rerun the original `sweep`
or `subset` invocation; the runner retries failed configurations. Both `perf`
and CUDA-enabled HPCToolkit must be installed on the host for CUDA runs.

Profiling is mandatory. The runner checks that `perf` and (for CUDA) the three
HPCToolkit tools (`hpcrun`, `hpcstruct`, `hpcprof`) are on `PATH`, and checks that
`perf stat` can access CPU counters before building. Loading HPCToolkit via
Spack does not install the separate `perf` command. On Rocky Linux, install
it with `sudo dnf install perf` and verify access with
`perf stat -e cycles:u,instructions:u -- true`. GPU profiling also needs a
CUDA-enabled host HPCToolkit installation in the same shell as the runner.
Use small `--prompt`, `--generate`, `--count`, and `--repetitions` settings when
testing on a laptop. Use a **new** `--results-dir` for an old timing-only plan:

```sh
python3 llama/llama_gpu_bench-random.py subset \
  --source /path/to/llama.cpp --model /path/to/model.gguf \
  --device cuda --cuda-arch 75 --count 1 \
  --results-dir /path/to/bench-results-profiled
```

Small local checks (including a tiny direct clang/CMake build):

```sh
python3 -m unittest discover -s llama -p 'test_*.py' -v
```
