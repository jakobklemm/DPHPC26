# GROMACS CUDA runs on RTX 5060 Ti, GPU code compiled by clang (D-INFK student cluster)

Theotime, 2026-11-26/27. 114 runs (32 sweep + 80 random + 2 end baselines), 0 failed,
~28 GPU-hours (test job 6278, sweep job 6290, random job 6354).

Same idea as `gb10-clang-*` (see `GB10_CLANG_README.md` and `clang-cuda/README.md`): clang 22
compiles the CPU **and** the GPU code, so our flags also reach the CUDA kernels. This is the
second GPU for the GROMACS clang runs (x86 host + desktop Blackwell GPU).

## Hardware and software

- RTX 5060 Ti (sm_120, 16 GB), host Intel Xeon E5-2680 v4 (Broadwell, AVX2).
  **3 CPUs per job** (the cluster gives 5060 Ti jobs 3 CPUs; the old nvcc runs in
  `5060ti-sweep` / `5060ti-subset150` had 6), so `-ntomp 3` and 3 build workers.
- Sweep on `studgpu-node09`, random on `studgpu-node01` (the cluster picks the node).
- clang 22.1.8 (official LLVM release, x86_64) for CPU and GPU code, CUDA 12.8, CCCL 3.4.3
  (CUDA's own `cub` does not compile with clang 22), CMake 3.31.6.
- GROMACS 2026.4 (same release tarball as all GROMACS runs), FFTW 3.3.10 built by GROMACS.
- No profiling (`--no-profile`).

## Benchmark

Same GPU benchmark as the old nvcc 5060 Ti runs (`5060TI_README.md`):

| | |
|---|---|
| Input | `water_0096.tpr` (96k atoms) |
| mdrun | GPU-resident (`-nb gpu -pme gpu -bonded gpu -update gpu`), `-ntmpi 1 -ntomp 3 -pin on -nsteps 50000 -resethway -notunepme -noconfout` |
| Runs | 1 warmup + 3 measured, `benchmark.json` = mean ns/day |

There is no CPU-only run in these jobs.

## Folders

| Folder | What | Runs |
|---|---|---|
| `5060ti-clang-sweep` | `sweep`: baseline + every other value alone, GPU flags included | 32 |
| `5060ti-clang-random80` | `random --count 80 --min-flags 1` (harmonic sampler, seed 20261021) | 80 |
| `5060ti-clang-baseline` | baseline build at the end of each job: sweep job (node09) and random job (node01) | 2 |

Same files per run as the other GROMACS folders (`build.log.gz`, `benchmark/*/md.log.gz`, ...).
`ml/load_gromacs.py` reads all 114 runs (0 excluded; same builds are merged into one row); the platform name is
the same as for the old nvcc 5060 Ti runs, and `gpu_compiler: clang` + the `cuda:*` options tell
them apart.

- Changed options per random build: k=1-3: 34, 4-6: 14, 7+: 32 (none is the baseline).
- 80 runs are 76 different builds (the sampler drew 4 a second time).

## Flags (`gromacs/flags-5060ti-clang.json`)

`flags-5060ti.json` (the nvcc 5060 Ti catalog) plus the clang `cuda_compiler` section of
`flags-clang.json`: `cuda:-O` (O3/O2/O1), `cuda:-finline-functions`, `cuda:-funroll-loops`,
`cuda:-ffp-contract` (fast/on/off). 20 options, sweep = 32 builds. First value = baseline.

## Test before the campaign (job 6278, not uploaded)

Baseline config, one build each: clang (CPU + GPU) 231.3 ns/day, nvcc 12.8 for the GPU code
242.2 (clang -4.5 %), clang + `cuda:-fno-inline-functions` 141.2 (-39 %). Energies clang vs
nvcc agree (conserved energy -1.25234e6 vs -1.25235e6).

## Main results

Baseline 230.8 ns/day (sweep start, node09), 230.6 (sweep end, node09), 223.1 (random end,
node01): no drift within a job, ~3 % between the two nodes.

Sweep (one change at a time, mean of 3 runs vs baseline):

| Change | ns/day | vs baseline |
|---|---|---|
| `-fno-inline-functions` (CPU) | 79.0 | **-66 %** |
| `cuda:-fno-inline-functions` | 141.1 | **-39 %** |
| `-Oz` | 195.0 | -15.5 % |
| `-Os` | 213.9 | -7.3 % |
| `-fno-unroll-loops` (CPU) | 216.7 | -6.1 % |
| `GMX_SIMD=None` | 217.0 | -6.0 % |
| `cuda:-fno-unroll-loops` | 222.3 | -3.7 % |
| `cuda:-ffp-contract=off` | 223.2 | -3.3 % |
| everything else (`-march`, LTO, `-O1`/`-O2`, `cuda:-O1`/`-O2`, fftpack, SIMD kernels, ...) | | within ±1.3 % |

- `-O2`, `-ffp-contract=fast` and `GMX_SIMD=AVX_256` show -4 to -5 % in the mean, but only
  because one of their 3 runs was slow (~200 vs ~230): no real effect.
- CPU no-inline costs much more here than on GB10 (0.34x vs 0.77x): with 3 slow Broadwell
  cores the CPU part of each step (launches, bookkeeping) becomes the bottleneck.
- CPU-side options (SIMD, fftpack) matter little: the run is GPU-resident.

Random (80 builds): 60.1-226.4 ns/day (x3.8), median 198.1; 11 % within 5 % of the median.
The two inline flags explain most of it:

| Random builds | n | median ns/day |
|---|---|---|
| neither CPU nor GPU no-inline | 44 | 220 (75 % within 5 %) |
| CPU `-fno-inline-functions` | 23 | 71 |
| `cuda:-fno-inline-functions` | 26 | 95 |
| both | 13 | 60-73 |

Noise: low. Median max/min of the 3 measured runs 0.2 %; 7 of 114 builds have > 5 % (always
one slow run out of 3).

Build time: median 589 s per random build (393-772 s), 675 s in the sweep, all on 3 cores.

## How to run

Student cluster, files in `clang-cuda/` (on the cluster in `clang-campaign/jobs/`):

```sh
sbatch -t 12:00:00 campaign_5060ti_gromacs_clang.sbatch gromacs-5060ti-clang-sweep sweep
sbatch -t 23:00:00 campaign_5060ti_gromacs_clang.sbatch gromacs-5060ti-clang-random random --count 80 --min-flags 1
```

The job runs the runner, then one baseline build into `<subdir>-baseline`
(`baseline_args.py` prints the profile arguments of sweep entry 0). Resumable: resubmit the
same command (fixed build path, because the runner stores `build_dir` in its settings).
`test_gromacs_5060ti_clang.sbatch` = the 3-build test above.
