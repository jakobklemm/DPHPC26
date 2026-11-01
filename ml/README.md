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
```

- Replace `path/to/system.json` with the target machine's metadata.
- `train`: save one model per application; `predict`: load it and output flags.
- `suggest`: fit fresh; rank unmeasured builds by performance + uncertainty.
- `evaluate`: compare all models; at least **two hardware platforms** per application.
- Default data: `llama/results/`, `gromacs/results/`; folders and ZIPs supported.
- Other data: add `--data path/to/runs.zip` to `train`, `suggest`, or `evaluate`.
- Application: `--app llama.cpp` or `--app gromacs`; omit for both.
- Saved models: `models/`; change with `--models path/to/models` **before** the command.
- Options/help: `python -m ml.main train --help` (likewise for other commands).

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
| `no-hardware` | Option-only ridge; test hardware features' contribution |
| `no-hardware-interactions` | Option-pair ridge without hardware inputs |
| `no-hardware-random-forest` | Option-only random forest |
| `no-hardware-catboost` | Option-only CatBoost |
| `no-hardware-gaussian-process` | Option-only Gaussian process |
| `no-hardware-kernel-ridge` | Option-only kernel ridge |
| `no-hardware-support-vector` | Option-only support vector regression |
| `no-hardware-small-mlp` | Option-only small MLP |
| `random` | Uniform valid choice; baseline, no `suggest` |

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
- Full selection: combinations of observed option values, including unmeasured combinations; potentially expensive.
- `predict --search local` (default): local search (`best_build`), seconds; exact for `hardware`/`no-hardware`, a local optimum for models where options interact
- `predict --search exhaustive`: checks every combination in batches (`exhaustive_best_build`), memory bounded, but slow (GB10 llama.cpp: ~35 min)
- `suggest` collects the full candidate list
- Training decision report (all models): local search around the best build, changing one or two options at a time; exact for `hardware`/`no-hardware`, a local optimum for the others
- CatBoost: included dependency; required for evaluation, imported lazily for other workflows.

| Change | Where (`ml/`) |
| --- | --- |
| Estimator defaults / model details | Individual files in `models/` |
| Shared encoding, fitting, uncertainty, selection | `models/common.py` |
| Available models | `models/__init__.py` → `MODEL_TYPES` |
| Commands / evaluation | `main.py` / `evaluate.py` |
| Data and hardware extraction | `load.py`, `load_llama.py`, `load_gromacs.py` |
| Build validity | `validation.py` + application loaders |

**Add a regressor:** create `models/<name>.py` → subclass `RegressionPerformanceModel` → implement `_make_regressor(seed, n_features)` → register in `MODEL_TYPES`. Shared workflows reused automatically.
