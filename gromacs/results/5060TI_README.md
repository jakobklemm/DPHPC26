# GROMACS CUDA runs on RTX 5060 Ti (D-INFK student cluster)

Theotime, 2026-11-10/13. 156 runs (28 sweep + 128 subset), 0 failed, ~52 GPU-hours.
Second GROMACS machine after GB10 (see `GB10_README.md`); same scripts, flags and input.

## Hardware and software

- Node `studgpu-node01`: NVIDIA GeForce RTX 5060 Ti (Blackwell, sm_120, 16 GiB GDDR7, driver 615.71.09)
  + Intel Xeon E5-2680 v4 (Broadwell, 2016, AVX2/FMA, no AVX-512). The job gets **6 cores** (one socket)
  and 24 GiB RAM; the node is shared with other users' jobs (own cores and GPU, shared memory bandwidth).
- clang 19.1.7 (official LLVM x86_64 release, links with lld), CUDA 12.8.93, CMake 3.28.3, Ubuntu 24.04.
- GROMACS 2026.4, FFTW 3.3.10 built by GROMACS, same `water_GMX50_bare` inputs as GB10.
- `perf` is blocked on this cluster (`perf_event_paranoid=4`): no CPU counters, `--no-profile` as on GB10.

## Two measurements per build (new)

Every build is benchmarked twice with the same `gmx` binary, 1 warmup + 3 measured runs each:

| File | mdrun | Input | Steps |
|---|---|---|---|
| `benchmark.json` | GPU-resident, as GB10: `-ntmpi 1 -ntomp 6 -pin on -nb gpu -pme gpu -bonded cpu -update gpu` | water 96k atoms | 50000 |
| `benchmark-cpu.json` | **CPU-only**: `-nb cpu -pme cpu -bonded cpu -update cpu`, `GMX_DISABLE_GPU_DETECTION=1` | water 48k atoms | 2000 |

Both use `-resethway -notunepme -noconfout`. Run settings are fixed for all builds; only build options change.
`ml/load_gromacs.py` reads `benchmark.json` (GPU) as before; `benchmark-cpu.json` has the same format
(plus `"mdrun_mode": "cpu_only"`) and is not read by the loader yet. Raw logs: `benchmark/` and `benchmark-cpu/`.

## Folders

| Folder | What | Runs |
|---|---|---|
| `5060ti-sweep` | `sweep`: baseline + every other value alone | 28 |
| `5060ti-subset150` | `subset --count 150` (5 options changed per build, seed 20261021), stopped at 128 | 128 |

The subset stopped at 128/150: from 2026-11-12 the cluster gives 5060 Ti jobs only 3 CPUs (was 6),
so the remaining builds would have run on a different setup. Two builds cut off by job time limits are left out.

## Flags (`gromacs/flags-5060ti.json`)

`flags.json` with two changes: `CMAKE_CUDA_ARCHITECTURES` = `120` only, `-flto=full` removed
(no effect on GB10, far too slow to build with 6 cores). The filter drops ARM-only values
(`ARM_SVE`, `ARM_NEON_ASIMD`, `-mcpu=*`) and `AVX_512`/`-march=x86-64-v4` (no AVX-512 on this CPU).
Baseline = `GMX_SIMD=AVX2_256`, `-O3`, `-march=x86-64`, everything else default.

## Main results

Baseline: GPU run 264.4 ns/day (GB10: 233), CPU-only run 18.55 ns/day.

Sweep (one change at a time, mean of 3 runs vs baseline):

| Change | GPU run | CPU-only run |
|---|---|---|
| `-fno-inline-functions` | **-55.0%** | **-95.7%** |
| `GMX_USE_SIMD_KERNELS=OFF` | -0.2% | **-82.5%** |
| `GMX_SIMD=None` | -3.8% | **-82.7%** |
| `GMX_SIMD=AVX2_128` / `AVX_256` / `SSE4.1` | ~0% | **-32.6% / -26.7% / -24.6%** |
| `GMX_FFT_LIBRARY=fftpack` | +0.1% | **-18.1%** |
| `-Oz` / `-Os` | **-10.0%** / -3.8% | **-13.7%** / -7.5% |
| `-fno-unroll-loops` | -3.1% | -5.9% |
| everything else (LTO thin, `-march`, `-Xptxas`, fp-contract, shared libs, ...) | within ±2% | within noise (see below) |

- The CPU-only run is where CPU build flags matter; in the GPU run only inlining and `-Oz/-Os` matter.
- Compared with GB10, the GPU run reacts more to CPU flags (no inlining -55% here vs -17% on GB10):
  6 old cores make the CPU part (neighbour search 11.5% of the time vs 2.4% on GB10) more important.

Subset (128 builds): GPU run 81.7-272.7 ns/day (median 253.4), CPU-only 0.6-18.4 ns/day (median 3.2).
Five options change per build, so per-flag medians of the subset are confounded; use the sweep or the model.

Noise: GPU run very stable (median max/min of the 3 runs 0.3%). CPU-only runs sometimes have one ~10% slow
run (13/128 subset builds > 5% spread, probably other jobs on the shared node), so the **median of the 3 runs**
is more robust than the mean for `benchmark-cpu.json`.

Build time: 5-9 min (median 7.2), ~6x GB10. Whole build incl. both benchmarks: 10-50 min (slowest with
`-fno-inline-functions`, whose CPU-only run is ~20x slower).

## How to run

```sh
python3 gromacs/gromacs_gpu_bench-random.py sweep --device cuda --cuda-arch 120 --no-profile \
  --flags-file gromacs/flags-5060ti.json --tpr gromacs/inputs/water_0096.tpr --nsteps 50000 \
  --cpu-tpr gromacs/inputs/water_0048.tpr --cpu-nsteps 2000 --results-dir <dir>
```

`gromacs/jobs/setup_5060ti.sbatch` (one-time check) and `campaign_5060ti.sbatch` are the exact SLURM
scripts used. `--cpu-tpr` is optional; without it the scripts behave as for the GB10 runs.
