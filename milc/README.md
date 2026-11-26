# MILC CPU benchmark

Builds and benchmarks MILC (`ks_imp_rhmc` / `su3_rhmd_hisq`, RHMC with HISQ quarks) for random flag
configurations. CPU only, x86-64 Linux with Docker; MILC is pinned inside the image
(commit 551dc479, develop) together with a thermalized 8^4 lattice, so runs need no network.
Run from the repository root:

```sh
docker pull ghcr.io/jakobklemm/dphpc26:milc-cpu
docker tag ghcr.io/jakobklemm/dphpc26:milc-cpu dphpc26:milc-cpu
# or build it: docker build -t dphpc26:milc-cpu -f milc/Dockerfile.cpu milc

mkdir -p ~/dphpc-data/milc
docker run -d --name milc-random -v ~/dphpc-data/milc:/data dphpc26:milc-cpu random \
  --count 500 --workers 16 --threads 8 --results-dir /data/results/random500
```

`--workers`: parallel compile jobs. `--threads`: OpenMP threads of the benchmark; use your number of
physical cores. `docker logs -f milc-random` shows progress, `docker stop` pauses and `docker start`
resumes. Values your CPU cannot run are filtered out (e.g. `x86-64-v4` without AVX-512), so plans
differ between CPUs with different features. `random --count 500 --dry-run --results-dir /tmp/x`
prints the plan; `sweep` changes one flag at a time. About 15 s per configuration on a Ryzen 7 5800X.

Sampling matches the llama run: seed 20261021, harmonic toggle counts, invalid configurations
resampled. Workload: 10 MD steps of one trajectory, 5 repetitions, timed per trajectory.

## Making sure the flags reach the binary

- Every build uses a fresh build directory, no ccache. Two separate baseline builds give the same
  code hash, so a hash difference comes from the flags.
- The catalog flags are appended last to every compile command, and each build is checked
  (`build.json`): CMake cache, preprocessor defines, compile commands, OpenMP linking, and a hash of
  the compiled code (for LTO, of the LTO object code).
- Values that produce identical code were removed: `KS_MULTICG` = `OFFSET`/`REVERSE`/`REVHYB`
  (no implementation, MILC silently runs `HYBRID`), `INLINE_C=ON` and `INLINE_SSE=ON` (macros this
  program never uses; the `_GLOBAL` versions work), and vector widths that are already the default
  for the chosen `-march`.
- **Blocked:** `PRECISION=1` (single) with `KS_MULTICG=FAKE`. MILC's solver goes NaN
  (`final_rsq= -nan`) and does not converge: 40 of 41 such runs failed. Each run is also checked
  for solver convergence, and records CG iterations and plaquettes.
- CPU cache: an 8^4 lattice fits in cache. On 12^4, memory bandwidth hid most flags (2% instead
  of 17% for `INLINE_C_GLOBAL`). Results therefore do not represent large lattices.
- Not used: the catalog's CUDA/QUDA options (no NVIDIA GPU on the machine that made these runs).

## Results (Ryzen 7 5800X, 8 threads; clang 18.1.3)

`results/ryzen5800x-sweep`: 81 builds, each flag alone and the compiler flags also with
`-march=x86-64-v3`, `native`, thin and full LTO. 0 failed. Run before the two inert vector-width
values were excluded, so the current sweep plan has 79.

`results/ryzen5800x-cpu-random-harmonic-500`: 500 configurations (237 distinct), 460 completed.
The 40 failures are all `PRECISION=1` + `FAKE` (run before that pair was blocked). The one such
run that completed had unstable CG iterations (7907, 7571, 6952, ...); leave it out of analysis.
Each distinct configuration has its own code hash. The most frequent configuration ran 187 times
with medians of 2.440-2.443 s in every block of 100 runs (no drift); repeated configurations agree
within 0.3% on average. All completed double precision runs give the same plaquette (to ~2e-11).

Effect of single changes on trajectory time (vs. baseline 2.44 s):

| Change | Speed |
|---|---|
| `PRECISION=1` | 1.24x |
| `-march=x86-64-v3` | 1.08x |
| `-flto=full` / `-flto=thin` | 1.03x / 1.02x |
| `-fno-slp-vectorize` | 0.92x |
| `-fno-strict-aliasing` | 0.88x |
| `KS_MULTICG=FAKE` | 0.85x |
| `INLINE_C_GLOBAL=OFF` | 0.84x |
| `OMP=OFF` (1 thread) | 0.22x |

`-fno-vectorize`, `-fno-math-errno` and `-fno-inline-functions` change the code but stay within
about 1%. The ML code has no MILC loader yet.
