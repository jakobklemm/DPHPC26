# llama.cpp CUDA runs on GB10 (D-INFK student cluster)

Theotime, 2026-10-28/29. 546 runs, 0 failed, ~18.5 GPU-hours.

## Hardware and software

- Node `studgpu-spark01`: NVIDIA GB10 (DGX Spark), GPU sm_121, aarch64 CPU
  (10 Cortex-X925 + 10 Cortex-A725), 112 GiB memory shared by CPU and GPU.
- clang 19.1.7 (official LLVM aarch64 release, links with lld), CUDA 13.1.1
  (lowest CUDA available for aarch64), HPCToolkit 2026.0.1.
- llama.cpp 62df1815b, model Qwen2.5-3B-Instruct Q4_K_M
  (sha256 `626b4a66...`, same as the threadripper runs).
- Benchmark: pp512 + tg128, 3 repetitions, all layers on GPU (`-ngl 99`).

## Folders

| Folder | What | Runs |
|---|---|---|
| `gb10-sweep` | `sweep` mode: one flag ON at a time | 19 |
| `gb10-sweep-extras` | all-OFF baseline, and `-O2`, `-Os`, `-Oz` (not reached by the sweep) | 4 |
| `gb10-subset500` | `subset --count 500` (7 flags ON, default seed) | 500 |
| `gb10-sweep-repeat` (+ `-extras`) | same sweep again, 17 h later, to measure noise | 23 |

Each run folder has `build.json` (config), `benchmark.json` (performance),
`gpu-metrics.json` (HPCToolkit totals for the whole run), `system.json`, and
`build.log.gz`.

## Differences from the threadripper runs

- **No CPU profiling** (`cpu-metrics.json` missing): `perf` is blocked on GB10
  (`perf_event_paranoid=4`). Run with `--skip-cpu-perf`.
- **No `hpcstruct`**: it never finishes on `libcublasLt` on aarch64. GPU metrics are
  run totals, which do not need it. Run with `--skip-hpcstruct`.
- The raw HPCToolkit `profiles/` folders (1.6 GB) are not committed; `build.log` is gzipped.

## Main results

Noise: sweep vs repeat differ by 1.0% (sd) on pp512, 0.3% on tg128, max 2.1%.

| Flag ON (rest OFF) | pp512 | tg128 |
|---|---|---|
| all OFF (baseline) | 5765 tok/s | 97.7 tok/s |
| `GGML_CUDA_FA` | +13.9% / +12.5% | +5.9% / +5.8% |
| `-finline-functions` | +5.5% / +4.7% | ~0 |
| `GGML_CUDA_GRAPHS` | ~+1% | +2.2% / +2.3% |
| `GGML_CUDA_FORCE_CUBLAS` | -28.6% / -27.8% | ~0 |
| all other flags | within ±2% | within ±1% |

(first sweep / repeat). Subset: pp512 4049-6942 tok/s, tg128 96.6-104.6 tok/s.
