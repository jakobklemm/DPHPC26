# Build performance models

- **Goal:** find fast build flags for llama.cpp and GROMACS on a new machine with as few benchmarks as possible.
- **Approach:** active sampling (benchmark → refit → pick the next build by predicted performance + uncertainty), optionally warm-started from other machines (transfer learning).
- **Inputs:** benchmark results + hardware metadata (`system.json`).
- **Offline comparison:** `simulate` replays sampling sessions on measured builds (vs. random, with/without pretraining).


## 1. Run active sampling (llama.cpp and GROMACS)

**Active sampling** finds a fast build for this machine with few benchmarks. Each step: the model proposes the next build (high predicted performance + high uncertainty) → build + benchmark it → refit the model on all results so far → repeat. The first `--initial` builds (default 2, the minimum a model needs) are random, since there is no model yet. The model learns from this machine's builds only.

**Transfer learning** (`--pretrained`) is active sampling with a head start from the other machines' benchmarks:

- A pretrained model is fitted once on the other machines and then kept fixed.
- It picks the first `--initial` builds (its top predictions) instead of random ones.
- The local model (`--model`) learns only the pretrained model's error on this machine
- Option values the other machines never had (e.g. AVX-512 flags on the i5, CUDA options for a CPU-only history) get no effect from the pretrained model. Only the local model can learn their effect. The evaluation CSVs (and the `simulate --output` CSV) count them per session in `unseen_values` (empty without pretraining); options that are constant on the machine do not count.

Both run in the container. The work folder `~/dphpc-data` on the host (mounted as `/data`) keeps the llama.cpp source, model weights, builds and results between runs; any other path works if used consistently.

```bash
# Once: build the image (from the repository root); one of:
docker build -f docker/llama.cpp.cuda.Dockerfile -t dphpc26:cuda .
docker build -f docker/llama.cpp.cpu.Dockerfile -t dphpc26:cpu .

# Or simply use the pre-built images on Github
# docker pull ghcr.io/jakobklemm/dphpc26:cuda (or tagged cpu or vulkan)

# Once: create the work folder
# Separate folder such that we dont mount a folder from the repository
mkdir -p ~/dphpc-data/llama # you may choose a different location in the commands

# Once: copy the llama flag catalog and historical benchmarks
cp -r llama/flags.json llama/results ~/dphpc-data/llama/

# Start a shell in the container; one of:
docker run --rm -it --gpus all --entrypoint bash -v ~/dphpc-data:/data dphpc26:cuda   # NVIDIA GPU, omit --gpus all if only a CPU is available
docker run --rm -it --entrypoint bash -v ~/dphpc-data:/data dphpc26:cpu               # CPU only

# Inside the container: prepare the work folder (once per work folder / machine)
runner setup --skip-train

# Inside the container: run a session
runner active-sampling --steps 10                             # cold start
runner active-sampling --steps 10 --pretrained hardware       # transfer learning from the other machines
```

- `runner setup` detects the hardware (`system.json`), clones llama.cpp and downloads the GGUF weights; a repeated run reuses them.
- `/data/llama/results/` must hold the historical benchmarks (flag values, pretraining).
- After code changes: rebuild the image; cached layers mean only `ml/` and `runner.sh` are copied again.

| Option | Meaning |
| --- | --- |
| `--steps N` | Builds to benchmark (required, maximum 25); failed builds count and are never proposed again |
| `--model` | Local model, refitted after every build; default `gaussian-process`; also `smac-forest`, `bayes-pairs`, `sparse-bayes`, `random` |
| `--pretrained` | Transfer learning: this model is fitted on the **other** machines in the results; `--model` then learns its error here. Default: none (cold start) |
| `--pretrain-on WORD…` | Pretrain only on platforms whose name contains one of these words |
| `--initial N` | Start builds before the local model takes over (default 2, minimum 2; counted in `--steps`): the pretrained model's top picks, or random without `--pretrained` |
| `--seed` | Start builds and candidate search (default 0) |
| `--device` | `cpu` or `cuda`; default: `cuda` if an NVIDIA GPU and the CUDA toolkit are found |
| `--output` | Results folder; default `/data/llama/results` |
| `--space` | JSON flag → allowed values, may include values never benchmarked; default: values in the results |

- Output: `<output>/active-<time>/` with one folder per build (loadable as training data), `session.json` and `best.json`.
- The last of the `--steps` builds is the model's final prediction, so it is measured too (if it was already built or failed, the last step samples normally). With `--model random` all builds are random.
- At the end: one result, the fastest of all builds measured in the session (also in `best.json`).
- Builds excluded by `flags.json` or invalid for the hardware are never proposed.

### GROMACS

`--app gromacs` runs the same session for GROMACS (`ml/live_gromacs.py`). Each build is
configured, compiled and benchmarked exactly like `gromacs/gromacs_gpu_bench-profile.py --no-profile`
(same CMake settings, `mdrun` arguments, 1 warmup + 3 measured runs, `benchmark.json`), so the
session folders load as training data next to the campaign runs. Performance is ns/day.

```bash
python -m ml.main active-sampling --app gromacs --steps 10 --pretrained hardware \
  --workdir /data --data /data/gromacs/results
```

There is one GROMACS benchmark, the same as in the GB10 and RTX 5060 Ti campaigns: CUDA builds
(`GMX_GPU=CUDA`, GPU architecture of this machine), GPU-resident `mdrun`
(`-nb gpu -pme gpu -bonded cpu -update gpu`), input `water_0096.tpr`, 50000 steps; the loader reads its
`benchmark.json` (the CPU-only `benchmark-cpu.json` of the 5060 Ti is not used). It is defined in one block,
"The benchmark", at the top of `ml/live_gromacs.py`; to switch to another benchmark, change that block and
measure the history the same way. GROMACS live sampling therefore runs on `--device cuda` only.

Work folder (`--workdir`, default `$DATA_DIR` or `/data`); the paths can be overridden:

| Path | Option | What |
| --- | --- | --- |
| `gromacs/gromacs-src/` | `--source` | GROMACS source tree (the campaigns used 2026.4) |
| `gromacs/fftw-3.3.10.tar.gz` | `--fftw` | FFTW tarball, built by GROMACS for `GMX_FFT_LIBRARY=fftw3` |
| `gromacs/inputs/water_0096.tpr` | `--tpr` | benchmark input, made once with `gromacs/prepare_input.sh`; must be the same input as in the history |
| `gromacs/flags.json` | — | flag catalog (read from `$DATA_DIR` if set) |
| `gromacs/results/` | `--output` | session results (default) |

- `--threads` (OpenMP threads of `mdrun` and build jobs, default: all available cores) is GROMACS-only.
- Compiler: clang/clang++; `cuda` also needs `nvcc` (on `PATH` or `CUDACXX`).
- `GMX_BUILD_OWN_FFTW` and `GMX_SIMD_ARM_SVE_LENGTH` are not tuning options: as in the harness, they follow from
  `GMX_FFT_LIBRARY` and from the CPU's SVE length. The loader leaves them out of the build configuration, so
  `predict`/`suggest` do not print them either. `CMAKE_CUDA_ARCHITECTURES` follows from the GPU, like for llama.cpp.
- A build that silently differs from the configuration (e.g. other SIMD reported by `gmx --version`) counts as failed.

## 2. Run other commands (outside the container)

### Setup

```bash
python3 -m venv .venv                    # Once
source .venv/bin/activate               # Each terminal
python -m pip install -r ml/requirements.txt
python -m pip install -r evaluation/requirements-notebook.txt   # Optional for evaluation/active-sampling-evaluation.ipynb (then select the .venv kernel)
```

### Available Commands

```bash
python -m ml.load                                                    # Inspect data
python -m ml.main train --model random-forest --app llama.cpp          # Fit + save
python -m ml.system --output system.json                            # Detect this machine's hardware (for --system)
python -m ml.main predict --model random-forest --app llama.cpp --system system.json
python -m ml.main suggest --model random-forest --app llama.cpp --system system.json -n 5
python -m ml.main evaluate --mode platform --app llama.cpp --random-trials 1000 --seed 42
python -m ml.main active-sampling --steps 10 --pretrained hardware --workdir ~/dphpc-data   # Live active sampling (section 1)
python -m ml.main simulate --app llama.cpp --steps 25                  # Active sampling on measured builds
python -m ml.main simulate --app llama.cpp --model sparse-bayes --pretrained no-hardware   # With transfer learning
```

| Command | Description | Required flags | Optional flags |
| --- | --- | --- | --- |
| `ml.load` | Print the loaded benchmark data | — | — |
| `train` | Fit one model per application and save it to `models/` | — | `--model`, `--app` (`llama.cpp` or `gromacs`; default both), `--data` (folders or ZIPs; default `llama/results/`, `gromacs/results/`) |
| `predict` | Load the saved model and output the best flags for a machine | `--system` (target machine's `system.json`) | `--model` (same as `train`), `--app`, `--search` (see below), `--space`, `--seed` |
| `suggest` | Fit fresh; list unmeasured builds worth benchmarking next (performance + uncertainty) | `--system` | `--model`, `--app`, `--data`, `-n` (count, default 5), `--search`, `--space`, `--seed` |
| `evaluate --mode platform` | Compare all models, holding out one machine at a time; needs **≥ 2 platforms** per application | — | `--app`, `--data`, `--random-trials` (random baseline repeats), `--seed` |
| `active-sampling` | Live active sampling on this machine (section 1) | `--steps` | see section 1 |
| `simulate` | Replay active sampling on each machine's measured builds, to compare models offline | — | `--model` (several allowed), `--steps`, `--initial`, `--seeds` (sessions per model), `--jobs`, `--pretrained` (adds `corrected-<model>` sessions), `--pretrain-on`, `--output` (CSV for plots), `--app`, `--data` |

- Options/help: `python -m ml.main train --help` (likewise for other commands).

### Search space (`suggest`, `predict`)

| Flag | `suggest` | `predict` |
| --- | --- | --- |
| `--search bounded` (default) | Random unmeasured combinations + flag edits around promising builds · benchmark flag values · up to 2,000 by default | Previous winners + random combinations + flag edits around winners and top predictions · saved model’s flag values · up to 10,000 total |
| `--search preset` | every combination of flag values seen in loaded benchmarks · invalid builds and builds already measured on the target excluded · no random sampling | — |
| `--search local` | — | one- or two-flag edits around the current best build · saved model’s flag values · variable count |
| `--search exhaustive` | — | All valid measured and unmeasured combinations · saved model’s flag values |
| `--space PATH` (bounded) | Random unmeasured combinations + flag edits around promising builds · JSON flag values · same budget | Eligible previous winners + random combinations + flag edits around winners and top predictions · JSON flag values · same budget |

## 3. Choose a model

`--model` picks the model for train, predict and suggest (default hardware) and active-sampling (default gaussian-process); predict needs the same one train saved. simulate can take several (--model a b …); evaluate defaults to all historical/local pairs, except that hardware local models with a `no-hardware-*` twin (and `hardware-three-way`) are skipped: the local model only sees one machine, so its hardware inputs are constant and these variants duplicate the no-hardware ones. Use --model and --pretrained to restrict (or explicitly re-add) them.


| `--model` | What / why |
| --- | --- |
| `hardware` | Bayesian ridge; regularized option × hardware effects |
| `hardware-interactions` | Ridge + option pairs; interacting flags |
| `hardware-three-way` | Ridge + option-pair × hardware; machine-dependent pair effects. Uses a kernel over benchmark rows to fit without expanding three-way features |
| `random-forest` | Nonlinear trees; automatic conditional interactions |
| `catboost` | Boosted trees; native categorical settings |
| `gaussian-process` | Kernel + native uncertainty; benchmark selection |
| `kernel-ridge` | Regularized nonlinear RBF regression |
| `support-vector` | Regularized RBF regression with error tolerance |
| `small-mlp` | Small neural network; nonlinear comparison |
| `smac-forest` | Random forest with tree spread as uncertainty; active sampling |
| `bayes-pairs` | Bayesian ridge on options + all option pairs; active sampling |
| `sparse-bayes` | Sparse Bayesian regression (ARD), prunes irrelevant options; active sampling |
| `no-hardware` | Option-only ridge; test hardware features' contribution |
| `no-hardware-interactions` | Option-pair ridge without hardware inputs |
| `no-hardware-random-forest` | Option-only random forest |
| `no-hardware-catboost` | Option-only CatBoost |
| `no-hardware-gaussian-process` | Option-only Gaussian process |
| `no-hardware-kernel-ridge` | Option-only kernel ridge |
| `no-hardware-support-vector` | Option-only support vector regression |
| `no-hardware-small-mlp` | Option-only small MLP |
| `random` | Uniform valid choice; baseline, no `suggest` |

- Uncertainty (the ± and the exploration in active sampling): Bayesian predictive spread for the ridge models, `gaussian-process`, `bayes-pairs`, `sparse-bayes`; spread between trees for `smac-forest`; spread of 8 bootstrap fits for `random-forest`, `catboost`, `kernel-ridge`, `support-vector`, `small-mlp`. Tree and bootstrap spreads are rough, uncalibrated heuristics.
- Ridge models and `bayes-pairs` fit in the training row space and retain prior
  uncertainty outside it. `hardware-three-way` computes feature dot products
  directly and uses a square matrix indexed by training builds. Other ridge
  variants use a reduced SVD of their feature matrices, which can still require
  substantial memory and computation for wide interaction matrices.
- `baseline` in evaluation: best-on-average measured configuration; no CLI model.
- `no-hardware-*` baseline uses the corresponding model's estimator defaults,
  constructor overrides, and uncertainty method, learning from build options only
  (hardware is still required for build validity and hardware-determined settings)

## 4. Read results

| Command | Output | How to read it |
| --- | --- | --- |
| `active-sampling` | tokens/s (llama.cpp) or ns/day (GROMACS) per build and a line per prediction round; at the end "best build (X tokens/s)"; the five evaluation CSVs in the session folder | The best build is the fastest of all measured builds, including the model's final prediction (built as the last step). Prediction scores in the round lines are model-native and may change scale on refit |
| `simulate` | Per model: `best@n`, `model_pick`, `rank_correlation`, `seconds` | `best@n`: best build after n builds, relative to the pool's best (**1.0** = found it). `model_pick`: final model's choice, same scale. `rank_correlation`: predicted vs. measured order on unmeasured builds. A model is only useful if it beats `random` |
| `evaluate` (`--mode simulate` / `live`) | Five CSV files in `--output` (`evaluation.final.csv`, `.quality.csv`, `.timings.csv`, `.convergence.csv`, `.improvement.csv`) | One row per session, round or attempt; see section 6 for what each file and column means |
| `evaluate --mode platform` | Per held-out machine: selection score per model, `error*` | Selection score: measured performance of the picked build / that machine's best (**1.0 best**). `error*`: mean absolute prediction error (**0 best**). All models choose from the same measured builds |
| `train` | Fit quality, then option costs per machine | Fit quality is on the training data only: ridge models (`hardware`, `no-hardware`, …) print "explains X% … estimated noise ±Y%", the others "training log-space R²" plus their uncertainty method. Costs (e.g. `-12% ±3%`): predicted change from switching the best build to another value, with `(with X=Y)` when another option must change too; exact for `hardware`/`no-hardware`, approximate for the others. `random` fits no model and prints no costs |

## 5. Evaluation: Active sampling with benchmarking

### Benchmark workload selection

A llama.cpp benchmark measures two phases: reading a prompt of N tokens (**pp**N, prompt processing) and generating N tokens (**tg**N, text generation). Tokens/s depends on these sizes, so runs with different sizes are not comparable.

- **Default:** every data-loading command uses only llama.cpp **pp512/tg128** runs. This also applies to the historical (pretraining) data for live sampling. GROMACS is not affected.
- **Live sampling** (`active-sampling`, `evaluate --mode live`) always uses pp512/tg128 and rejects other values, so the historical data matches what is benchmarked live.

```bash
python -m ml.main evaluate --app llama.cpp --prompt 512 --generate 128 \
  --model hardware random --pretrained none --steps 25 --output evaluation-pp512-tg128
```

**Filter:** a run is kept only if it has exactly one prompt-only and one generation-only phase of the requested sizes. Filtering happens before averaging and normalization, and the loader prints the included/excluded run counts per platform.

**Current data:** all data is used except the Intel i5 sweep (pp32/tg32) and the CPU-only Threadripper runs (`threadripper_cpu_*`, pp128/tg1024).

### Metrics

`evaluate` runs whole sampling sessions and reports how good the model's choice is after every round. A session is one sampling run of one policy on one target platform.

| `--mode` | What it does |
|---|---|
| `simulate` (default) | Replays sessions offline on the already measured builds of each platform (the *pool*). No building or benchmarking |
| `live` | Runs one real ative sampling session on this machine: build + benchmark each attempt, as in section 1. Requires exactly one `--model` and one `--pretrained` (`none` for no pretraining); `random` cannot correct a historical model |
| `platform` | The former evaluation: holds out one machine at a time and compares models (section 4) |

```bash
python -m ml.main evaluate --app llama.cpp --steps 25 --output evaluation-results        # all policies, simulated
python -m ml.main evaluate --app llama.cpp --model hardware --pretrained none --seeds 1  # small smoke run
python active-sampling/run_all_policies.py                                               # same evaluation, plus its replay CSV
```

**Options** (`evaluate`; `run_all_policies.py` takes the same ones where they exist)
- `--model`: local models (default: all). `--pretrained`: pretrained models (default: all plus `none`).
- `--steps`: budget of benchmark attempts, at most 25 (default 25). `--initial`: startup builds (default 2).
- `--seeds`: repeats of every policy, cold and historical (default 3; 3–5 is enough for now). Seed `s` (from `--seed` on) sets the random start builds and the `random_state` of the historical and local model fits, so stochastic models (forests, MLPs, bootstraps) vary between seeds; ridge models are deterministic and repeat identically. The seed is recorded in the `seed` column. `--jobs`: parallel sessions; use `--jobs 1` for timings without concurrent workers.
- `--output`: CSV folder (default `evaluation-results`; live: parent of the session folder). Use a fresh path for this CSV format.
- `--prompt` / `--generate` (llama.cpp), `--data`, `--app`, `--pretrain-on`; live only: `--system`, `--device`, `--workdir`, `--space`; GROMACS live only: `--source`, `--fftw`, `--tpr`, `--threads`.

**Policies, budget and rounds**
- **Cold start** (`pretrained = none`): the local model starts from scratch. Runs `--seeds` times, because the start builds are random.
- **Historical** (`pretrained = X`, local model shown as `corrected-<model>`): model X is fitted on all other platforms (the target is held out), and the local model learns its error on the target. Runs `--seeds` times, with X refitted per seed. Needs at least two platforms.
- **Budget:** at most 25 attempts, counting the 2 startup builds and failed builds.
- **Rounds:** `step` counts all attempts; `round` starts at 1 with the first attempt after startup. The standalone `active-sampling` command reports the first fitted prediction as round 0.
- Malformed JSON runs are skipped with a warning.

**The five CSV files.** Each row starts with `application`, `platform`, `pretrained`, `model`, `seed`, `workload`.

| File | One row per | Answers | Key columns |
|---|---|---|---|
| `evaluation.final.csv` | session | How good is the result at the end? | `final_build`, `final_actual_performance` vs `pool_best_performance`, `final_is_pool_best`, `first_pool_best_round` (first time the best build is picked), `stable_pool_best_round` (from when it stays on it) |
| `evaluation.quality.csv` | round | How good is the current recommendation? | `recommended_percent_of_pool_best`, `recommended_percent_of_best_observed`, `recommended_prediction_absolute_error`, `attempted_build`, timing columns |
| `evaluation.timings.csv` | attempt (incl. failed) | What was tested, what is the best found so far, what did it cost? | `attempted_measured_performance`, `best_observed_performance`, `best_observed_percent_of_pool_best`, `attempt_status`, timing columns |
| `evaluation.convergence.csv` | round | Does the recommendation still change? | `changed_from_previous`, `is_pool_best`, `same_as_final` |
| `evaluation.improvement.csv` | round | How far does the recommendation move from the first pick, and does it pay off? | `flags_changed_from_first`, `flags_changed_from_final`, `prediction_gain`, `actual_gain_from_first`, `actual_fraction_of_best` |

**Reading the columns**
- **Recommended vs. attempted:** `recommended_*` is the model's current pick (what you would take if you stopped now). `attempted_*` / `attempt_*` is the build benchmarked in that round. quality.csv rates the pick; timings.csv rates the best build measured so far, which is the fair comparison with `random`.
- **Performance:** in simulation, `*_percent_of_pool_best` compares with the best build in the pool (100 = found it). In live mode there is no pool, and `recommended_predicted_performance` is a predicted tokens/s, not a measurement.
- **Time (seconds):** `prediction_seconds`, `next_sample_seconds` and `refit_seconds` are the model's own cost per round. `initialization_seconds` is a one-time startup cost, repeated in every row. `session_elapsed_seconds` is the running total. Live only: `attempt_*_seconds` time configure, compile and benchmark of the attempted build (blank in simulation).
