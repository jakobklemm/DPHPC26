# GROMACS CPU-only runs on Apple M1 Max (ARM, macOS)

Theotime, 2026-11-24/26. 177 runs (4 baseline + 23 sweep + 150 random), 0 failed, ~18 h build + benchmark time.
ARM CPU measurements (TA request), the GROMACS twin of `llama/results/M1MAX_CPU_README.md`.
The GB10 runs are ARM too, but GPU-bound; here only the CPU runs, so CPU build flags matter a lot.

## Hardware and software

- MacBook Pro, Apple M1 Max: 8 performance + 2 efficiency cores, 32 GB, macOS 26.7.3, on AC power
  (runs on battery / Low Power Mode were thrown away and rerun, see below).
- Apple clang 21.0.0 (`/usr/bin/clang`, not an LLVM release as on the cluster) + Homebrew libomp
  (`OpenMP_ROOT=$(brew --prefix libomp)`, CMake finds it as `-Xclang -fopenmp`), CMake 4.4.4, Unix Makefiles.
- GROMACS 2026.4, FFTW 3.3.10 built by GROMACS (`GMX_BUILD_OWN_FFTW`), `GMX_GPU=OFF`.
- No profiling (`--no-profile`, as all other GROMACS runs).

## Benchmark

Same CPU-only benchmark as `benchmark-cpu.json` of the 5060 Ti runs (`5060TI_README.md`), so the two CPUs
can be compared directly:

| | |
|---|---|
| Input | `water_0048.tpr` (48k atoms, same file, sha256 `2da18113...`) |
| mdrun | `-ntmpi 1 -ntomp 8 -pin on -nb cpu -pme cpu -bonded cpu -update cpu -nsteps 2000 -resethway -notunepme -noconfout` |
| Runs | 1 warmup + 3 measured, `benchmark.json` = mean ns/day |

8 threads = the 8 performance cores. `-pin on` does nothing on macOS (no thread affinity; GROMACS prints no
warning there). Here the CPU-only result is `benchmark.json` (no GPU run), so `ml/load_gromacs.py` reads it
as usual: platform `Apple M1 Max`, 177 runs -> 120 rows, 0 invalid.

## Folders

| Folder | What | Runs |
|---|---|---|
| `m1max-cpu-baseline` | baseline build (every option at its first value) 2x at the start, 2x at the end | 4 |
| `m1max-cpu-sweep` | `sweep`: baseline + every other value alone | 23 |
| `m1max-cpu-random150` | `random --count 150 --min-flags 1` (harmonic sampler, seed 20261021) | 150 |

Same files per run as the other GROMACS folders (`build.log.gz`, `benchmark/*/md.log.gz`, ...).
`random`: k=1-3: 98 builds, 4-6: 30, 7+: 22; 150 runs are 116 different builds (sampler repeats).

## Flags (`gromacs/flags.json`, filtered for this CPU)

16 options. The filter now reads CPU features on macOS from `sysctl hw.optional` (mapped to the Linux
names) and offers `-mcpu` values only if the CPU has their instructions: `-mcpu=neoverse-v2` and
`-mcpu=cortex-x925` need SVE (the M1 has none, the binary would crash), so they are dropped here.
Filtered catalogs for GB10 and the 5060 Ti node are unchanged.

- `GMX_SIMD`: `ARM_NEON_ASIMD` (baseline), `None`
- `-mcpu`: `-mcpu=generic` (baseline), `-mcpu=neoverse-n1`, `-mcpu=native` (= apple-m1)
- all other options as on GB10 (`GMX_DOUBLE`, SIMD kernels, accuracy bits, fftpack, shared libs,
  `-O`, vectorize, SLP, unroll, inline, strict aliasing, fp-contract, math-errno, `-flto` thin/full)

## Main results

Baseline 31.6 ns/day (sweep start), 31.2 / 28.7 (campaign start), 31.5 / 31.3 (campaign end):
no drift over the campaign. The 5060 Ti node's Xeon (6 cores) does 18.6 ns/day on the same benchmark.

Sweep (one change at a time, mean of 3 runs vs baseline), next to the 5060 Ti CPU-only run:

| Change | M1 Max | 5060 Ti node CPU-only |
|---|---|---|
| `-fno-inline-functions` | **-95%** | -95.7% |
| `GMX_SIMD=None` | **-63%** | -82.7% |
| `GMX_USE_SIMD_KERNELS=OFF` | **-62%** | -82.5% |
| `GMX_DOUBLE=ON` | **-44%** | (not in sweep) |
| `-Oz` / `-Os` | **-42%** / -8% | -13.7% / -7.5% |
| `GMX_FFT_LIBRARY=fftpack` | -11% | -18.1% |
| `-fno-unroll-loops` | -7% | -5.9% |
| `GMX_SIMD_ACCURACY_BITS_SINGLE=16` | +4% | |
| everything else (`-mcpu`, LTO thin/full, `-O1`/`-O2`, fp-contract, aliasing, shared libs, ...) | within ±2% | within noise |

- Same main effects as on x86. SIMD off costs less than on the Xeon (NEON is 128-bit, AVX2 256-bit),
  `-Oz` costs much more (-42% vs -14%).
- `-mcpu=native` / `neoverse-n1` bring nothing (probably because the hot GROMACS kernels are hand-written NEON SIMD code).

Random (150 builds): 1.05-33.7 ns/day (x32), median 11.8; only 17% of builds within 5% of the median,
so the builds spread widely, they do not cluster near the baseline.

Noise: low. Median max/min of the 3 measured runs 0.9%; 15 of 177 builds have > 5% (Mac in use at times).

Build time: 90 s - 49 min (median 2.6 min; `-flto=full` and `-fno-inline-functions` are the slowest).

Runs on battery: the Mac was unplugged once (2026-11-25 10:18), macOS then switches to Low Power Mode.
The one build measured then (random 71) was moved out and rerun on AC; in `random-executions.json`
its first attempt has status `discarded`. All runs here are on AC power.

## How to run

```sh
export OpenMP_ROOT=$(brew --prefix libomp)
python3 gromacs/gromacs_gpu_bench-random.py random --device cpu --no-profile --count 150 --min-flags 1 \
  --source <gromacs-2026.4> --fftw <fftw-3.3.10.tar.gz> --tpr water_0048.tpr --nsteps 2000 \
  --threads 8 --workers 10 --results-dir <dir>
```

`sweep` the same without `--count/--min-flags`. Baseline repeats: `gromacs_gpu_bench-profile.py` with
the baseline configuration (`sweep` plan entry 0). Mac on AC power, Low Power Mode off, `caffeinate -i -s`.
