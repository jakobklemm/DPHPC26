# Build performance models

- **Goal:** find fast build flags for llama.cpp and GROMACS on a new machine with as few benchmarks as possible.
- **Approach:** active sampling (benchmark → refit → pick the next build by predicted performance + uncertainty), optionally warm-started from other machines (transfer learning).
- **Inputs:** benchmark results + hardware metadata (`system.json`).
- **Offline comparison:** `simulate` replays sampling sessions on measured builds (vs. random, with/without pretraining).


## 1. Run active sampling (currently only for llama)

**Active sampling** finds a fast build for this machine with few benchmarks. Each step: the model proposes the next build (high predicted performance + high uncertainty) → build + benchmark it → refit the model on all results so far → repeat. The first `--initial` builds (default 2, the minimum a model needs) are random, since there is no model yet. The model learns from this machine's builds only.

**Transfer learning** (`--pretrained`) is active sampling with a head start from the other machines' benchmarks:

- A pretrained model is fitted once on the other machines and then kept fixed.
- It picks the first `--initial` builds (its top predictions) instead of random ones.
- The local model (`--model`) learns only the pretrained model's error on this machine

Both run in the container. The work folder `~/dphpc-data` on the host (mounted as `/data`) keeps the llama.cpp source, model weights, builds and results between runs; any other path works if used consistently.

```bash
# Once: build the image (from the repository root); one of:
docker build -t dphpc26:cuda .                        # NVIDIA GPU (also runs CPU-only)
docker build -f Dockerfile.cpu -t dphpc26:cpu .       # CPU only, smaller

# Once: create the work folder
# Separate folder such that we dont mount a folder from the repository
mkdir -p ~/dphpc-data/llama # you may choose a different location in the commands

# Once: Copy the flag catalog and the historical benchmarks in this folder (from the repository root)
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
| `--steps N` | Builds to benchmark (required); failed builds count and are never proposed again |
| `--model` | Local model, refitted after every build; default `gaussian-process`; also `smac-forest`, `bayes-pairs`, `sparse-bayes`, `random` |
| `--pretrained` | Transfer learning: this model is fitted on the **other** machines in the results; `--model` then learns its error here. Default: none (cold start) |
| `--pretrain-on WORD…` | Pretrain only on platforms whose name contains one of these words |
| `--initial N` | Start builds before the local model takes over (default 2, minimum 2; counted in `--steps`): the pretrained model's top picks, or random without `--pretrained` |
| `--seed` | Start builds and candidate search (default 0) |
| `--device` | `cpu` or `cuda`; default: `cuda` if an NVIDIA GPU and the CUDA toolkit are found |
| `--output` | Results folder; default `/data/llama/results` |
| `--space` | JSON flag → allowed values, may include values never benchmarked; default: values in the results |

- Output: `<output>/active-<time>/` with one folder per build (loadable as training data), `session.json` and `best.json`.
- At the end: the best **measured** build and the model's pick (may be unmeasured).
- Builds excluded by `flags.json` or invalid for the hardware are never proposed.

## 2. Run other commands (outside the container)

### Setup

```bash
python3 -m venv .venv                    # Once
source .venv/bin/activate               # Each terminal
python -m pip install -r ml/requirements.txt
```

### Available Commands

```bash
python -m ml.load                                                    # Inspect data
python -m ml.main train --model random-forest --app llama.cpp          # Fit + save
python -m ml.system --output system.json                            # Detect this machine's hardware (for --system)
python -m ml.main predict --model random-forest --app llama.cpp --system system.json
python -m ml.main suggest --model random-forest --app llama.cpp --system system.json -n 5
python -m ml.main evaluate --app llama.cpp --random-trials 1000 --seed 42
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
| `evaluate` | Compare all models, holding out one machine at a time; needs **≥ 2 platforms** per application | — | `--app`, `--data`, `--random-trials` (random baseline repeats), `--seed` |
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

`--model` picks the model for train, predict and suggest (default hardware) and active-sampling (default gaussian-process); predict needs the same one train saved. simulate can take several (--model a b …); evaluate always compares all.


| `--model` | What / why |
| --- | --- |
| `hardware` | Bayesian ridge; regularized option × hardware effects |
| `hardware-interactions` | Ridge + option pairs; interacting flags |
| `hardware-three-way` | Ridge + option-pair × hardware; machine-dependent pair effects. **Currently fails on llama.cpp** (~59k features, LAPACK overflow) |
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
- `baseline` in evaluation: best-on-average measured configuration; no CLI model.
- `no-hardware-*` baseline uses the corresponding model's estimator defaults,
  constructor overrides, and uncertainty method, learning from build options only
  (hardware is still required for build validity and hardware-determined settings)

## 4. Read results

| Command | Output | How to read it |
| --- | --- | --- |
| `active-sampling` | tokens/s per build; at the end "best measured" and "model pick (predicted X of best measured)" | **Best measured** is the safe answer (actually measured). The model pick may be unmeasured; X > 1: the model expects it to be faster |
| `simulate` | Per model: `best@n`, `model_pick`, `rank_correlation`, `seconds` | `best@n`: best build after n builds, relative to the pool's best (**1.0** = found it). `model_pick`: final model's choice, same scale. `rank_correlation`: predicted vs. measured order on unmeasured builds. A model is only useful if it beats `random` |
| `evaluate` | Per held-out machine: selection score per model, `error*` | Selection score: measured performance of the picked build / that machine's best (**1.0 best**). `error*`: mean absolute prediction error (**0 best**). All models choose from the same measured builds |
| `train` | R², then option costs per machine | R²: fit on the training data only. Costs (e.g. `-12% ±3%`): predicted change from switching the best build to another value; exact for `hardware`/`no-hardware`, approximate for the others |

## 5. Edit code

| Change | Where (`ml/`) |
| --- | --- |
| Estimator defaults / model details (affects all commands) | Individual files in `models/` (`_make_regressor()`) |
| Shared encoding, fitting, uncertainty, selection | `models/common.py` |
| Available models | `models/__init__.py` → `MODEL_TYPES` |
| Commands / evaluation | `main.py` / `evaluate.py` |
| Active sampling (`ask`, `tell`, `recommend`) and its simulation | `active_sampling.py` |
| Live llama.cpp session: build, benchmark, device detection, results folders | `live_llama.py` |
| Transfer learning (`--pretrained`): pretrained model + local error correction | `models/residual_correction.py` |
| `no-hardware-*` variants and the `evaluate` baseline | `models/baseline_models.py` |
| Hardware detection (`system.json`) | `system.py` |
| Shared candidate generation, scoring budgets, and default seed | `search.py` |
| Data and hardware extraction | `load.py`, `load_llama.py`, `load_gromacs.py` |
| Build validity; exclude builds/flag values | `validation.py` + application loaders; exclusions in `llama/flags.json`, `gromacs/flags.json` (repository root) |

**Add a regressor:** create `models/<name>.py` → subclass `RegressionPerformanceModel` → implement `_make_regressor(seed, n_features)` → register in `MODEL_TYPES`. Shared workflows reused automatically.
