# llama.cpp CUDA runs on GB10, GPU code compiled by clang (D-INFK student cluster)

Theotime, 2026-11-19. 322 runs (22 sweep + 300 random), 0 failed, ~9.6 GPU-hours (job 4619).

## Why these runs

In all earlier CUDA runs (`gb10-*`, threadripper, 5060 Ti) our compiler flags only reached
the CPU code: the runner passes them as `CMAKE_C/CXX_FLAGS_RELEASE`, which clang uses, but
the GPU code (`.cu`) is compiled by nvcc with fixed flags (`-DCMAKE_CUDA_FLAGS=-lineinfo`).
So the GPU kernels were the same in every build, apart from the CMake options
(`GGML_CUDA_FA`, `FORCE_MMQ`, ...), and the GPU results "clump".

Here clang 22 compiles the GPU code too (`--cuda-compiler clang`), and the GPU flags
(`cuda:*` below) are sampled like the CPU flags. See `clang-cuda/README.md`.

## Hardware and software

- Node `studgpu-spark01`: NVIDIA GB10 (DGX Spark), GPU sm_121, aarch64 CPU
  (10 Cortex-X925 + 10 Cortex-A725), same node as `gb10-subset500`.
- clang 22.1.8 (official LLVM aarch64 release, lld) for CPU **and** GPU code, CUDA 13.1.1
  (headers, libraries, ptxas), CMake 3.31.6 (3.28 cannot use clang as CUDA compiler).
  GPU arch `sm_121a`. Setup: `clang-cuda/env-gb10-clang.sh`.
- llama.cpp 62df1815b (as `git archive`, so `source_git` is empty in build.json) with
  `llama/clang-cuda.patch`: clang gets the fast-math flags nvcc uses
  (`-ffast-math -fno-finite-math-only -fcuda-flush-denormals-to-zero -fgpu-approx-transcendentals`),
  and cub is switched off for clang-CUDA only (CUDA 13.1's cub does not compile with clang;
  llama.cpp then uses its own HIP fallback kernels, not in llama-bench's hot path).
- Model Qwen2.5-3B-Instruct Q4_K_M (sha256 `626b4a66...`, same as all llama runs).
- Benchmark: pp512 + tg128, 3 repetitions, `-ngl 99`. No profiling (no `gpu-metrics.json`).

## Folders

| Folder | What | Runs |
|---|---|---|
| `gb10-clang-sweep` | `sweep`: baseline + every value alone, GPU flags included | 22 |
| `gb10-clang-random300` | `random --count 300 --min-flags 1` (Jakob's harmonic sampler, seed 20261021) | 300 |

Same files per run as `gb10-subset500`: `source.yml`, `system.json`, `build.json`,
`benchmark.json`, `benchmark.log`, `build.log.gz`.

- `random --min-flags 1`: number of changed flags k drawn with weight 1/(k+1), but never 0
  (with k=0, ~29 % of the builds would be the baseline again).
  Result: k=1-3: 150 builds, 4-6: 57, 7+: 93 (old `subset`: always 7).
- 300 runs are 237 different builds: the sampler drew 63 builds a second time (mostly k=1,
  where only ~25 builds exist). `ml/load.py` averages them, like repeated runs.
- The sweep baseline (first run) has `GGML_CPU_REPACK=ON`, the random baseline `OFF`.

## Flags (`llama/flags-clang.json`)

`llama/flags.json` plus a `cuda_compiler` section; first value = baseline:

- `cuda:-O`: `-O3`, `-O2`, `-O1`
- `cuda:-finline-functions`: `-finline-functions`, `-fno-inline-functions`
- `cuda:-funroll-loops`: `-funroll-loops`, `-fno-unroll-loops`

(`-ffp-contract` left out for llama: the fixed `-ffast-math` would override it.)
`build.json` has `cuda_compiler: "clang"` and `cuda_flags` (the GPU flags passed);
`ml/load_llama.py` reads them as `gpu_compiler` and `cuda:<group>` options. nvcc runs
have neither, so these options are `default` (= nvcc) there.

## Results (vs. the old nvcc runs `gb10-subset500`)

| | old nvcc: slowest / fastest | clang | old: builds within 5 % of median | clang |
|---|---|---|---|---|
| pp512 | x1.7 | x9.7 | 36 % | 35 % |
| tg128 | x1.08 | x1.8 | 100 % | 59 % |

GPU flags now change GPU speed. Biggest effects on pp512 (median relative to all builds):
`cuda:-fno-inline-functions` 0.17x, `cuda:-fno-unroll-loops` 0.36x, `GGML_CUDA_FORCE_MMQ=ON`
0.38x; on tg128: `cuda:-fno-inline-functions` 0.65x. Results still form groups, because a
few on/off flags have a big effect.

Toolchain check (same config, test job before the campaign): clang -O3 GPU code is ~4-5 %
slower than nvcc on pp512, ~1 % on tg128; `test-backend-ops` MUL_MAT / SOFT_MAX / ROPE / RMS_NORM pass.

## Not changed

`ml/live_llama.py` still builds with nvcc: live sampling cannot reproduce `gpu_compiler=clang`
configs yet.
