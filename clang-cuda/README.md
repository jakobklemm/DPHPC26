# clang for the GPU code (CUDA)

**Problem:** with nvcc our compiler flags only reach the CPU code. The runners pass them in
`CMAKE_C/CXX_FLAGS_RELEASE`; nvcc reads `CMAKE_CUDA_FLAGS`, which stayed fixed (`-lineinfo`,
plus `-Xptxas -O` for GROMACS). So the GPU kernels were identical in every build and GPU
runtimes clump.

**Fix:** let clang compile the CUDA code too (`--cuda-compiler clang`). The default stays
`nvcc`, and nvcc plans are byte-identical to before.

## Test on GB10 (sm_121, CUDA 13.1), same config, GPU code only changed

| GPU build | llama pp512 | llama tg128 | GROMACS ns/day |
|---|---|---|---|
| nvcc (old setup) | 5699 | 99.4 | 240 |
| clang -O3 | 5481 (-4%) | 98.3 (-1%) | 233 (-3%) |
| clang -O1 | 1156 (-79%) | 96.6 (-2%) | 225 (-3.5%) |
| clang + -fno-inline-functions | 900 (-84%)* | 57.2 (-42%)* | 131 (-44%) |

\* llama: old all-OFF baseline flags (no-inline, no-unroll, no-vectorize, ...) on the GPU code.
Correctness: llama test-backend-ops (MUL_MAT/SOFT_MAX/ROPE/RMS_NORM) GPU vs CPU all OK;
GROMACS energies match nvcc (~1e-4).

## What changed

- `llama/` and `gromacs/` runners + profiles: `--cuda-compiler nvcc|clang`. With clang the
  flag file's `cuda_compiler` section is varied too (GPU flags, first value = baseline):
  `cuda:-O` (O3/O2/O1), `cuda:-finline-functions`, `cuda:-funroll-loops`
  (+ `cuda:-ffp-contract` for GROMACS; llama's fixed fast-math would override it).
  Catalogs: `llama/flags-clang.json`, `gromacs/flags-clang.json`.
- GROMACS got the llama `random` sampler (harmonic toggle count, weights 1/(k+1)).
- Both samplers: `--min-flags` (default 0 = unchanged; with 1 no build is an exact baseline
  repeat, otherwise ~29% are).
- GROMACS `build.json` `options` gets `gpu_compiler: clang` and the `cuda:*` names.
  **`ml/validation.py` does not know these names yet.**

## Toolchain (what is needed and why)

- **clang 22.1.8** (official LLVM release). clang 19 does not know sm_120/121 and supports
  CUDA only up to 12.5.
- **CMake 3.31.6.** 3.28 cannot identify clang as a CUDA compiler.
- CMake's CUDA check builds for a default arch that CUDA 13 dropped, so the scripts pass
  `--cuda-gpu-arch=sm_XX` themselves.
- **llama.cpp:** apply `llama/clang-cuda.patch`. The ggml CUDA CMake only knows nvcc flags.
  CUDA 13.1's `cub` headers do not compile with clang 22, so the existing HIP fallback is used
  for argsort/sum/mean/cumsum/top-k/ssm-scan; these are not in the llama-bench path.
- **GROMACS:** `-DGMX_CLANG_CUDA=ON`. It needs `cub`, so use **CCCL v3.4.3** (header-only,
  github.com/NVIDIA/cccl) via `--cccl`, with FP4/6/8 off. OpenMP for CUDA sources is given
  explicitly.
- nvcc with clang 22 as host compiler needs `-allow-unsupported-compiler`.

## Files here (student cluster paths)

- `env-gb10-clang.sh`: toolchain environment for GB10 jobs.
- `campaign_gb10_clang.sbatch <llama|gromacs> <subdir> <sweep|random> [args]`: one run.
- `chain_gb10_clang.sbatch <llama|gromacs> <count>`: sweep, then random in one job (same node).
- `env-5060ti-clang.sh`: toolchain environment for RTX 5060 Ti jobs (x86_64, clang 22.1.8,
  CUDA 12.8). The `cuda/12.8` module adds a non-existent `lib/`; `libcudart.so.12` is in `lib64/`.
- `campaign_5060ti_clang.sbatch <subdir> <sweep|random> [args]` or `... chain <count>`: llama on
  one 5060 Ti (sweep, then random). The cluster ignores `--nodelist`/`--exclude`, so jobs land on
  the always-on node `studgpu-node01` (shared with other users).
- `campaign_5060ti_gromacs_clang.sbatch <subdir> <sweep|random> [args]`: GROMACS on one 5060 Ti
  (catalog `gromacs/flags-5060ti-clang.json`), then one baseline build into `<subdir>-baseline`.
  Results: `gromacs/results/5060TI_CLANG_README.md`.
- `baseline_args.py <flags-file> <cuda-arch> [<option> <value> ...]`: prints the profile
  arguments of the sweep baseline (sweep entry 0), optionally with some options changed.
- `test_gromacs_5060ti_clang.sbatch`: 3-build test (clang baseline, nvcc baseline, clang +
  `cuda:-fno-inline-functions`) with build time, ns/day and energies.

On the cluster these files are in `clang-campaign/jobs/` and the runners in
`clang-campaign/{llama,gromacs}/`, so paths in the job scripts point there.

## llama-bench threads (`--threads`)

The llama runner now takes `--threads N` (passed to `llama-bench -t`, stored in the run settings
only when used). Without it, llama-bench starts one thread per core of the whole machine. That
is fine when the job owns the machine (GB10: 20 threads, 20 cores), but a 5060 Ti job gets only
3 of 28 cores. With `GGML_OPENMP=OFF`, ggml's own thread pool busy-waits, and 28 threads on
3 cores dropped tg128 from ~155 to ~70 tok/s with samples spread up to 40%. With `-t 3`, the
results are stable again (spread < 0.5%). Builds with `GGML_OPENMP=ON` were stable either way.
**Any team run where the job has fewer cores than the machine should pass `--threads`.**

Other GPUs: AMD -> ROCm/HIP (hipcc is clang, so flags reach the GPU code). Vulkan and Metal
compile shaders with their own compilers (see team chat for proposals).
