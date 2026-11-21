# GROMACS CUDA runs on GB10, GPU code compiled by clang (D-INFK student cluster)

Theotime, 2026-11-20/21. 183 runs (33 sweep + 150 random), 0 failed, ~21.9 GPU-hours (job 4620).

## Why these runs

In the earlier CUDA runs (`gb10-*`, `5060ti-*`) our compiler flags only reached the CPU
code; nvcc compiled the GPU code with fixed flags (`-lineinfo -Xptxas=-O*`), so the GPU
kernels were nearly the same in every build. Here clang 22 compiles the GPU code too
(`--cuda-compiler clang`) and the GPU flags (`cuda:*`) are sampled like the CPU flags.
Same idea as `llama/results/GB10_CLANG_README.md`; setup in `clang-cuda/README.md`.

## Hardware and software

- Node `studgpu-spark01`: NVIDIA GB10 (DGX Spark), GPU sm_121, aarch64 CPU
  (10 Cortex-X925 + 10 Cortex-A725), same node as `gb10-subset250`.
- clang 22.1.8 (official LLVM aarch64 release, lld, libomp) for CPU **and** GPU code
  (`-DGMX_CLANG_CUDA=ON`), CUDA 13.1.1, CMake 3.31.6, GPU arch `sm_121a`.
- CUDA 13.1's own CCCL (cub/thrust) does not compile with clang 22, so CCCL 3.4.3 from
  GitHub is used (`--cccl`), with `-DCCCL_DISABLE_NVFP4/6/8_SUPPORT`.
  OpenMP for CUDA set by hand (`OpenMP_CUDA_FLAGS=-fopenmp=libomp`, LLVM's libomp).
- GROMACS 2026.4, input `water_0096`, benchmark, mdrun options and metric (ns/day, mean of
  3 measured runs after 1 warmup) exactly as in `GB10_README.md`.

## Folders

| Folder | What | Runs |
|---|---|---|
| `gb10-clang-sweep` | `sweep`: baseline + every other value alone, GPU flags included | 33 |
| `gb10-clang-random150` | `random --count 150 --min-flags 1` (harmonic sampler, as llama) | 150 |

Same files per run as `gb10-subset250` (`build.log.gz`, `benchmark/*/md.log.gz`, ...).
On the cluster the folders were called `gromacs-gb10-clang-*`.

- `random` (new for GROMACS, Jakob's llama sampler): number of changed options k drawn
  with weight 1/(k+1), at least 1. Result: k=1-3: 69 builds, 4-6: 31, 7+: 50
  (old `subset`: always 5-7). 150 runs are 127 different builds (23 drawn twice).

## Flags (`gromacs/flags-clang.json`)

`gromacs/flags.json` with the nvcc `-Xptxas -O` option replaced by clang GPU flags
(first value = baseline):

- `cuda:-O`: `-O3`, `-O2`, `-O1`
- `cuda:-finline-functions`: `-finline-functions`, `-fno-inline-functions`
- `cuda:-funroll-loops`: `-funroll-loops`, `-fno-unroll-loops`
- `cuda:-ffp-contract`: `-ffp-contract=fast`, `-ffp-contract=on`, `-ffp-contract=off`

`build.json` `options` has `gpu_compiler: "clang"` and the `cuda:*` options, read by
`ml/load_gromacs.py` like the other options (nvcc runs: `default`).

## Results (vs. the old nvcc runs `gb10-subset250`)

| | old nvcc | clang |
|---|---|---|
| slowest / fastest build | x1.46 | x2.64 (86.9-229.3 ns/day) |
| builds within 5 % of the median | 67 % | 49 % |

- Sweep (one option changed): most options change ns/day by at most 3 %. Big effects:
  `cuda:-fno-inline-functions` 0.57x, CPU `-fno-inline-functions` 0.77x,
  CPU `-fno-vectorize` 0.90x.
- In the random runs, combinations go down to ~0.4x (e.g. GPU `-O1`/`-O2` or CPU `-Oz`
  together with other changes), so options interact.
- Toolchain check (test job before the campaign): clang `-O3` is ~3 % slower than nvcc;
  energies agree with nvcc to ~1e-4, conserved energy identical.
- Noise: ~1 % within a job, up to ~6 % between jobs on the same node.

## Not changed

`ml/live_gromacs.py` still builds with nvcc.
