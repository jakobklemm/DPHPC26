# GROMACS CUDA runs on GB10 (D-INFK student cluster)

Theotime, 2026-11-03/05. 309 runs (30 sweep + 250 subset + 29 repeat sweep), 307 OK, 2 failed, ~28 GPU-hours.

## Hardware and software

- Node `studgpu-spark01`: NVIDIA GB10 (DGX Spark), GPU sm_121, aarch64 CPU
  (10 Cortex-X925 + 10 Cortex-A725, SVE 128-bit), 112 GiB memory shared by CPU and GPU.
  Same node and toolchain as the llama.cpp GB10 runs.
- clang 19.1.7 (official LLVM aarch64 release, links with lld), CUDA 13.1.1, CMake 3.28.3.
- GROMACS 2026.4 (sha256 `58eda609...`), FFTW 3.3.10 built by GROMACS (`GMX_BUILD_OWN_FFTW`).
- Input: `water_GMX50_bare` 0096 (96k atoms, PME), made with `gromacs/prepare_input.sh`
  and `gromacs/bench.mdp`. The `.tpr` sha256 is in every `build.json` (`tpr_sha256`).
- Benchmark: `gmx mdrun -ntmpi 1 -ntomp 20 -pin on -nb gpu -pme gpu -bonded cpu -update gpu
  -nsteps 50000 -resethway -notunepme -noconfout`, 1 warmup + 3 measured runs.
  Everything runs on the GPU, like llama.cpp with `-ngl 99`.
  (`-bonded cpu` because water has no bonded interactions; `-bonded gpu` is a fatal error.)
- Performance = mean ns/day of the 3 measured runs (`benchmark.json` `mean`, higher is better).

## Folders

| Folder | What | Runs |
|---|---|---|
| `gb10-sweep` | `sweep` mode: baseline + every other value alone | 30 |
| `gb10-subset250` | `subset --count 250` (5 options changed per build, seed 20261021) | 250 |
| `gb10-sweep-repeat` | same sweep again, 1.5 days later, to measure noise (without `121-virtual`) | 29 |

Same layout as llama: each run folder has `source.yml`, `system.json`, `build.json`
(config; `options` is the flat dict that `ml/load_gromacs.py` reads), `benchmark.json`,
`build.log.gz`, and `benchmark/<warmup|measured><n>/{md.log.gz,mdrun.log.gz}` (raw GROMACS logs).
`*-plan.json`, `*-executions.json`, `*-settings.json` are the runner's own files (resume works from them).

## Flags (`gromacs/flags.json`)

First value of every option = baseline (a normal GROMACS Release build); the others are alternatives.

- CMake: `GMX_SIMD` (ARM_SVE / ARM_NEON_ASIMD / None), `GMX_USE_SIMD_KERNELS`,
  `GMX_SIMD_ACCURACY_BITS_SINGLE` (22 / 16), `GMX_FFT_LIBRARY` (fftw3 / fftpack), `BUILD_SHARED_LIBS`,
  `CMAKE_CUDA_ARCHITECTURES` (121 / 120), `GMX_CUDA_NB_SINGLE_COMPILATION_UNIT`.
  `GMX_DOUBLE` is fixed OFF (CUDA builds are mixed precision only).
- clang: `-O3/-O2/-O1/-Os/-Oz`, `-mcpu=generic/neoverse-n1/neoverse-v2/cortex-x925/native`,
  `-f[no-]vectorize`, `-f[no-]slp-vectorize`, `-f[no-]unroll-loops`, `-f[no-]inline-functions`,
  `-f[no-]strict-aliasing`, `-ffp-contract=on/fast/off`, `-f[no-]math-errno`, `-fno-lto/-flto=thin/-flto=full`.
- nvcc: `-Xptxas -O3/-O2/-O1`.
- Left out on purpose: `-ffast-math`/unsafe math (changes results), MPI, thread-MPI, profiling counters.

## Differences from the llama.cpp runs

- **No profiling** (`--no-profile`): no `cpu-metrics.json`, no `gpu-metrics.json`.
- `ml/validation.py` does not know the new option names yet (e.g. `GMX_USE_SIMD_KERNELS`, `-O`, `-mcpu`),
  so `valid()` rejects these runs. `load_table` reads them fine (307 runs → 278 rows, because the repeat sweep is averaged into the
matching sweep builds);
  `search`/`suggest`/`evaluate` will filter them out until the schema is updated.

## Main results

Noise: median run-to-run sd 0.7%. Same config in sweep vs repeat: sd 0.9%, mean -0.6%
(leaving out 2 configs with one slow run). About 5% of builds have one measured run much slower than
the other two (e.g. `[106, 196, 193]`), which looks like a node hiccup, not the flags; the median of
the 3 runs is more robust than the mean for these.

Sweep (one change at a time; first sweep / repeat; baseline 233.2 / 236.0 ns/day):

| Change | first sweep | repeat |
|---|---|---|
| `-fno-inline-functions` | **-17.1%** | **-18.2%** |
| `-Xptxas=-O1` | -1.9% | -2.9% (also -2.9% in the subset: small but probably real) |
| `-ffp-contract=off` | -8.9% (one slow run, sd 17%) | -0.6% → noise |
| `-fno-strict-aliasing` | +0.7% | -10.5% (one slow run, sd 19%) → noise |
| everything else | +0.1% to +2.6% | -2.4% to +0.6% |
| `CMAKE_CUDA_ARCHITECTURES=121-virtual` | failed | not run: driver too old for CUDA 13.1 PTX JIT, removed from `flags.json` |

The first sweep's baseline was measured first and came out a bit low, so "everything else" looks
slightly positive there; in the repeat it is around zero.

Subset (250 builds): 164.9-239.9 ns/day, median 231.8.

| | builds | best of 3 runs: min / median / max |
|---|---|---|
| inlining on (default) | 174 | 225 / 236 / 241 |
| `-fno-inline-functions` | 75 | 168 / 197 / 220 |

All other values change the median by 3% or less (`-Xptxas=-O1` -2.9%, `-Oz` -2.7%, `GMX_SIMD=None` -2.7%).
The GPU does almost all the work, so CPU build flags matter little here (a CPU-only test build showed
much bigger effects: `GMX_SIMD=None` -63%, no inlining -95%).

Failed: 1 subset build (`-Oz` + `-flto=thin` + `-fno-math-errno` + `-mcpu=generic`, run
`20261104T145828877108Z`): it compiles, but the simulation starts at 1.8e8 K, energies are NaN and mdrun
stops. It failed the same way twice, so it is a real miscompile, not noise. All OK runs start at the same
temperature (297.29 K).

`-flto=full` builds take 17-70 min (median build 80 s).

## How to run

Defaults expect the GROMACS 2026.4 source unpacked in `gromacs/gromacs-src` and
`gromacs/fftw-3.3.10.tar.gz` (from fftw.org, md5 `8ccbf6a5ea78a16dbc3e1306e234cc5c`); both are git-ignored.

```sh
gromacs/prepare_input.sh <path/to/gmx> 0096 gromacs/inputs/water_0096.tpr
python3 gromacs/gromacs_gpu_bench-random.py sweep --device cuda --cuda-arch 121 --no-profile \
  --tpr gromacs/inputs/water_0096.tpr --nsteps 50000 --results-dir <dir>
python3 gromacs/gromacs_gpu_bench-random.py subset --count 250 ...   # same options
```

`gromacs/jobs/*.sbatch` are the exact SLURM scripts used (paths are for the student cluster).
The scripts reuse `llama/` for shared helpers, so keep both folders side by side.
