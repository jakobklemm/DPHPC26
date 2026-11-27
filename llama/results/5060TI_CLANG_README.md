# llama.cpp CUDA runs on RTX 5060 Ti, GPU code compiled by clang (D-INFK student cluster)

Theotime, 2026-11-24/25. 140 runs (29 sweep + 111 random), 0 failed, ~27 GPU-hours
(job 5393: sweep + random, job 6015: more random).

Same scripts, flag catalog and model as `gb10-clang-*` (see `GB10_CLANG_README.md` and
`clang-cuda/README.md`), but on x86 with a desktop Blackwell GPU. This is the second hardware
where our flags also reach the GPU code.

## Hardware and software

- Node `studgpu-node01`: NVIDIA GeForce RTX 5060 Ti (16 GB GDDR7, sm_120, driver 615.71.09),
  host Intel Xeon E5-2680 v4 (Broadwell, AVX2, no AVX-512). Same node as the GROMACS
  `5060ti-*` runs. The node is shared: a job gets 1 GPU, **3 CPU cores** and 24 GB RAM;
  other users' jobs run next to it.
- clang 22.1.8 (official LLVM x86_64 release, lld) for CPU **and** GPU code, CUDA 12.8
  (same as the GROMACS 5060 Ti runs), CMake 3.31.6. GPU arch `sm_120a`.
  Setup: `clang-cuda/env-5060ti-clang.sh`, job script `clang-cuda/campaign_5060ti_clang.sbatch`.
- llama.cpp 62df1815b + `llama/clang-cuda.patch` (same tree as the GB10 clang runs).
- Model Qwen2.5-3B-Instruct Q4_K_M (sha256 `626b4a66...`, same as all llama runs).
- Benchmark: pp512 + tg128, 3 repetitions, `-ngl 99`, **`-t 3`** (see below).
  No profiling (`perf` is blocked on the node).

## Folders

| Folder | What | Runs |
|---|---|---|
| `5060ti-clang-sweep` | `sweep`: one value changed at a time, GPU flags included | 29 |
| `5060ti-clang-random` | `random --count 400 --min-flags 1 --threads 3` (harmonic sampler, seed 20261021), stopped by the job time limits (jobs 5393 + 6015) | 111 |

Files per run as in `gb10-clang-*`: `source.yml`, `system.json`, `build.json`,
`benchmark.json`, `benchmark.log`, `build.log.gz`.

- Changed flags per random build: k=1-3: 59, 4-6: 21, 7+: 31.
- 111 runs are 101 different builds (the sampler drew 10 a second time); `ml/load.py` averages them.
- `--count 400` is only an upper limit; the run stops at the job time limit. Job 6015 resumed
  job 5393's plan (same settings). Two builds cut off by the limits are left out.
- The sweep's first run is the reference build with `GGML_OPENMP=ON`; all other sweep runs
  and the random baseline have `GGML_OPENMP=OFF`.

## Important: `--threads 3`

llama-bench's default is one thread per core of the **whole machine** (28 here), but the job
owns only 3 cores. With `GGML_OPENMP=OFF`, ggml's own thread pool busy-waits, and 28 threads on
3 cores dropped tg128 from ~155 to ~70 tok/s, with the 3 samples up to 40 % apart. Test
(same build, alternating): `-t 3` gives 151-156 tok/s, the default 62-80. Builds with
`GGML_OPENMP=ON` were stable either way. The runner therefore got `--threads N` (passed as
`llama-bench -t N`, stored in the run settings). The first, noisy attempt was deleted.
GB10 runs were not affected: the job owns the whole node (20 threads = 20 cores).
**Team runs where the job gets fewer cores than the machine has should pass `--threads`.**

## Results

| | GB10 clang | 5060 Ti clang |
|---|---|---|
| pp512, typical build (median) | 5262 tok/s | 5886 tok/s (+12 %) |
| tg128, typical build (median) | 93.4 tok/s | 158.5 tok/s (+70 %) |
| pp512 slowest / fastest | x9.8 | x10.2 |
| tg128 slowest / fastest | x1.8 | x2.3 |

tg128 is memory-bound: GDDR7 (~448 GB/s) vs the GB10's shared LPDDR5x (~273 GB/s).

Sweep, one flag changed (ratio to the median sweep build, pp512 / tg128):

| Flag | GB10 | 5060 Ti |
|---|---|---|
| `cuda:-fno-inline-functions` | 0.17 / 0.65 | 0.15 / 0.50 |
| `cuda:-fno-unroll-loops` | 0.34 / 1.00 | 0.41 / 0.95 |
| `GGML_CUDA_FORCE_CUBLAS=ON` | 0.77 / 1.00 | 0.90 / 1.00 |
| `GGML_CUDA_FA=ON` | 1.10 / 1.06 | 1.07 / 1.08 |
| CPU `-finline-functions` | 1.03 / 1.00 | 1.13 / 1.02 |
| all other single flags | 0.96-1.03 | 0.96-1.01 |

The same flags matter on both GPUs; the size of the effect differs. CPU ISA flags
(AVX, FMA, `-march`, ...) do nothing: the whole model runs on the GPU.

## Noise

- Within a build, the 3 samples are 0.3 % apart (median).
- 18 of 111 random builds have a **low first pp512 sample** (e.g. 4575 / 5973 / 5978), probably
  because the GPU clocks down during the ~10 min build and llama-bench's warm-up is too short.
  tg128 is not affected. The team metric (`ml/load_llama.py`, mean time of pp + tg) changes by
  < 1 %, because tg128 is ~90 % of the time. **For pp512-only analysis use the median sample.**

## Build time

Median 620 s per random build (max 3170 s, LTO), 485 s in the sweep (many incremental builds):
3 Broadwell cores compile everything, including all CUDA kernels. The benchmark itself takes
~15 s. That is why ~27 GPU-hours give 140 runs here vs 322 on GB10.
