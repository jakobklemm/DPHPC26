# Build performance models

- **Purpose:** recommend build flags for llama.cpp and GROMACS; choose useful next benchmarks.
- **Inputs:** benchmark results + hardware metadata (`system.json`).
- **Workflow:** observed option values → valid combinations → predicted performance → best flags.

## 1. Set up

Run from the repository root:

```bash
python3 -m venv .venv                    # Once
source .venv/bin/activate               # Each terminal
python -m pip install -r ml/requirements.txt
```

## 2. Run

```bash
python -m ml.load                                                    # Inspect data
python -m ml.main train --model random-forest --app llama.cpp          # Fit + save
python -m ml.main predict --model random-forest --app llama.cpp --system path/to/system.json
python -m ml.main suggest --model random-forest --app llama.cpp --system path/to/system.json -n 5
python -m ml.main evaluate --app llama.cpp --random-trials 1000 --seed 42
python -m ml.main simulate --app llama.cpp --budget 25                  # Active sampling on measured builds
python -m ml.main simulate --app llama.cpp --model sparse-bayes --pretrained no-hardware   # With transfer learning
```

- Replace `path/to/system.json` with the target machine's metadata.
- `train`: save one model per application; `predict`: load it and output flags.
- `suggest`: fit fresh; rank unmeasured builds by performance + uncertainty.
- `evaluate`: compare all models; at least **two hardware platforms** per application.
- `simulate`: replay active-sampling sessions per machine (see `active_sampling.py`); `--output sessions.csv` for plots.
- `simulate --pretrained`: adds `corrected-<model>` sessions; the pretrained model is fitted on the other machines (`--pretrain-on` limits which) and `<model>` learns its error here.
- Default data: `llama/results/`, `gromacs/results/`; folders and ZIPs supported.
- Other data: add `--data path/to/runs.zip` to `train`, `suggest`, `evaluate`, or `simulate`.
- Application: `--app llama.cpp` or `--app gromacs`; omit for both.
- Saved models: `models/`; change with `--models path/to/models` **before** the command.
- Options/help: `python -m ml.main train --help` (likewise for other commands).

### Search space

| Flag | `suggest` | `predict` |
| --- | --- | --- |
| No search flag | Bounded search · random unmeasured combinations + flag edits around promising builds · benchmark flag values · up to 2,000 by default | Bounded search · previous winners + random combinations + flag edits around winners and top predictions · saved model’s flag values · up to 10,000 total |
| `--search bounded` | Random unmeasured combinations + flag edits around promising builds · benchmark flag values · up to 2,000 by default | Previous winners + random combinations + flag edits around winners and top predictions · saved model’s flag values · up to 10,000 total |
| `--search preset` | every combination of flag values seen in loaded benchmarks · invalid builds and builds already measured on the target excluded · no random sampling | — |
| `--search local` | — | one- or two-flag edits around the current best build · saved model’s flag values · variable count |
| `--search exhaustive` | — | All valid measured and unmeasured combinations · saved model’s flag values |
| `--space PATH` (bounded) | Random unmeasured combinations + flag edits around promising builds · JSON flag values · same budget | Eligible previous winners + random combinations + flag edits around winners and top predictions · JSON flag values · same budget |

- **Saved configurations only** · `simulate` replay pool
- **Bounded flag edits** · changes to 1–4 flags per build · one- and two-flag changes favored
- **JSON flag values** · explicitly allowed values, including values absent from training data
- **Previous winners** · fastest measured builds saved by `train` · target platform first · retraining for older model files
- **All combinations** · product of per-flag value counts before validity filtering

## 3. Choose a model

Use the same `--model` for training and prediction. Default: `hardware`.

| `--model` | What / why |
| --- | --- |
| `hardware` | Bayesian ridge; regularized option × hardware effects |
| `hardware-interactions` | Ridge + option pairs; interacting flags |
| `hardware-three-way` | Ridge + option-pair × hardware; machine-dependent pair effects |
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

- Active sampling: fit on one machine's builds only, no pretraining; `gaussian-process` also fits this role.
- Transfer learning: `ResidualCorrection` (`models/residual_correction.py`) keeps a pretrained model fixed and corrects it with a new machine's builds; any model above can take either role.
- `baseline` in evaluation: best-on-average measured configuration; no CLI model.
- `no-hardware-*` baseline uses the corresponding model's estimator defaults,
  constructor overrides, and uncertainty method, learning from build options only
  (hardware is still required for build validity and hardware-determined settings)

```bash
python -m ml.main train --model no-hardware-random-forest --app llama.cpp
python -m ml.main predict --model no-hardware-random-forest --app llama.cpp --system path/to/system.json
```

## 4. Change parameters

- **CLI defaults:** edit `_make_regressor()` in the relevant `ml/models/` file; retrain.
- **Experiment in Python:** constructor overrides; save for subsequent CLI prediction.

```python
from pathlib import Path
from ml.load import app_options, load_table
from ml.models import RandomForestPerformanceModel

model = RandomForestPerformanceModel(
    max_depth=6, min_samples_leaf=3,  # Limit overfitting
    ensemble_size=4,                # Fewer bootstrap fits; faster training (minimum 2)
    random_state=42,                # Repeatable fits in the same environment
).fit(app_options(load_table(app="llama.cpp")))

Path("models").mkdir(exist_ok=True)
model.save("models/llama.cpp.random-forest.joblib")
```

- Non-GP new models: full-data fit + 8 bootstrap fits by default; whole machines resampled, or rows with one machine.
- GP: one fit with native uncertainty. Bootstrap spread for other new models: **uncalibrated exploration heuristic**.
- Saved overrides apply to `predict`; `train` and `suggest` use current constructor defaults.

## 5. Read results / edit code

- Selection score: measured performance / platform's best measured build; **1.0 best**.
- `error*`: mean absolute prediction error; **0 best**. Training R²: fit diagnostic only.
- Evaluation: whole-machine holdout; identical eligible **measured** candidates for all models.
- Validity checks: hardware/toolchain compatibility; separate from predictive accuracy.
- Exhaustive prediction timing (GB10 llama.cpp): ~35 min.
- Training decision report (all models): local search around the best build, changing one or two options at a time; exact for `hardware`/`no-hardware`, a local optimum for the others
- CatBoost: included dependency; required for evaluation, imported lazily for other workflows.

| Change | Where (`ml/`) |
| --- | --- |
| Estimator defaults / model details | Individual files in `models/` |
| Shared encoding, fitting, uncertainty, selection | `models/common.py` |
| Available models | `models/__init__.py` → `MODEL_TYPES` |
| Commands / evaluation | `main.py` / `evaluate.py` |
| Active sampling (`ask`, `tell`, `recommend`) and its simulation | `active_sampling.py` |
| Shared candidate generation, scoring budgets, and default seed | `search.py` |
| Data and hardware extraction | `load.py`, `load_llama.py`, `load_gromacs.py` |
| Build validity | `validation.py` + application loaders |

### Live sampling, prediction, and simulation

- **Validation:** sampling, suggestions, and prediction share application-specific checks and `flags.json` exclusions, including historical llama rules. Candidates are filtered before scoring and checked before selection; unknown hardware capabilities still require build verification. Add exclusions to the application's `flags.json`
- **Live sampling** uses `ask()` to propose an unmeasured build from the allowed flag space. Benchmark it on the target and pass the result to `tell()`. After the budget, `recommend()` returns the fastest measured build. Pass `pretrained` to enable local correction of a historical model.
- **Live prediction** ranks candidates by predicted performance. Bounded `predict` scores previous winners and fresh random builds, then explores flag edits around the winners and top-scoring candidates. Its pick may be unmeasured.
- **Simulation** replays sampling within a fixed pool of previously benchmarked builds. It reveals each result after selection, so it compares sampling policies on that pool. Live candidate generation and new flag combinations require separate validation.

**Add a regressor:** create `models/<name>.py` → subclass `RegressionPerformanceModel` → implement `_make_regressor(seed, n_features)` → register in `MODEL_TYPES`. Shared workflows reused automatically.
