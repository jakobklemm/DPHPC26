# llama.cpp CPU-only runs on Apple M1 Max (ARM)

Theotime, 2026-11-22/23. 320 runs, 0 failed. These are the ARM CPU measurements the TA
asked for: the GB10 llama runs use an ARM host but are GPU-bound (`--device cuda`), and
all other CPU-only llama data is x86.

## Hardware and software

- MacBook Pro, Apple M1 Max: arm64, 8 performance + 2 efficiency cores, 32 GB.
  llama-bench uses 8 threads. CPU features: dotprod and fp16 yes; i8mm, bf16, SVE, SME no.
- macOS 26.7.3, Apple clang 21.0.0, Homebrew libomp 22.1.3 for OpenMP
  (`OpenMP_ROOT=$(brew --prefix libomp)`).
- llama.cpp 62df1815b, model Qwen2.5-3B-Instruct Q4_K_M
  (sha256 `626b4a66...`, same as all other llama runs).
- Benchmark: pp512 + tg128, 3 repetitions, `--device cpu`, no GPU layers.

## Folders

| Folder | What | Runs |
|---|---|---|
| `m1max-cpu-baseline` | all-OFF baseline: 2 before the sweep, 2 after the random runs (drift check) | 4 |
| `m1max-cpu-sweep` | `sweep` mode: one flag ON at a time | 16 |
| `m1max-cpu-random300` | `random --count 300 --min-flags 1` (seed 20261021), 251 unique configs | 300 |

Each run folder has `build.json` (config), `benchmark.json` (performance), `benchmark.log`,
`system.json`, `source.yml` and `build.log.gz`. No `cpu-metrics.json`: run with `--no-profile` (no `perf`
on macOS).

## Differences from the x86 CPU runs

- **ARM CPU levels use a new `-mcpu` compiler group** instead of x86 `-march`: baseline
  `-mcpu=generic` (plain ARMv8-A, like `-march=x86-64`), then `-mcpu=native`,
  `generic+dotprod`, `generic+dotprod+fp16` (`+i8mm`, `+sve` exist in `flags.json` but are
  removed by the filter because the M1 lacks them). Not `-march`: clang ignores
  `-mcpu=native` (added by `GGML_NATIVE`) as soon as any `-march` is given.
- **`GGML_CPU_KLEIDIAI`** (ARM KleidiAI kernels) is a new cmake flag, ARM only.
- x86 flags (`GGML_AVX*`, `GGML_FMA`, ...) are fixed OFF by the filter, as on any non-x86 host.
- `GGML_METAL`, `GGML_BLAS`, `GGML_ACCELERATE` are forced OFF on macOS (they default to
  ON there), so the runs are pure ggml CPU code.
- `--min-flags 1`: no all-OFF configs in the random set (the baseline folder has them).
  This avoids the ~1/3 baseline repeats the harmonic sampler draws otherwise.

## Main results

Baseline (mean of 4): pp512 34.3 tok/s, tg128 26.3 tok/s. Start vs end baseline: within 2%.

| Flag ON (rest OFF) | pp512 | tg128 |
|---|---|---|
| `-mcpu=native` | +104% | +74% |
| `GGML_NATIVE` | +102% | +69% |
| `-finline-functions` | +73% | +64% |
| `GGML_CPU_KLEIDIAI` | +26% | -13% |
| `-funroll-loops` | +9% | +7% |
| `GGML_LLAMAFILE` | +1% | -18% |
| all other flags | within ±4% | within ±11% (noise, see below) |

Random 300: pp512 18.8-265.4 tok/s (x14), tg128 12.5-51.4 tok/s (x4).
Best pp512 (265 tok/s): NATIVE + LTO + KleidiAI + REPACK + LLAMAFILE, `-O1`, inline,
unroll, slp-vectorize, unsafe-math, `-ffp-contract=fast`.

Coverage: 16 options vary (491,520 configs); every pair of option values appears together
at least twice. Flags ON per config: 1-3: 133, 4-6: 62, 7: 15, 8-16: 90.

## Noise (important)

The laptop was not idle during the whole campaign. 27 configs were measured more than once:
same-config sd is 7.9% on pp512 and 11.4% on tg128, max deviation 28%. This is much higher
than on GB10 (~1%). Treat flag effects below ~10% as noise.

Discarded and re-measured (not in this folder):
- random executions 289-300 and 2 end baselines ran on battery (macOS Low Power Mode),
  ~25% slower. They were re-run on AC; old attempts are listed as `discarded` under
  `previous_attempts` in `random-executions.json`.
- one run interrupted by a manual stop (2026-11-23 00:45), re-run on resume.
