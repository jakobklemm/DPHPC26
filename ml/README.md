# Build performance models

Uses local llama.cpp and GROMACS benchmarks to learn build performance, recommend
build options for a machine, suggest new benchmarks, and compare against baselines.

## Commands

Run from the repository root. Set up once; activate `.venv` in each new terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

```bash
python -m ml.load                                             # Inspect benchmark data
python -m ml.main train                                       # Train and save models
python -m ml.main predict --system path/to/system.json         # Recommend build options
python -m ml.main suggest --system path/to/system.json         # Suggest builds to benchmark
python -m ml.main evaluate --random-trials 1000 --seed 42       # Compare models and baselines
```

Data defaults to `llama/results` and `gromacs/results`, including folders and ZIPs.
Models are saved in `models/`. Replace `path/to/system.json` with the target
machine's metadata. `predict` loads saved models; `suggest` trains fresh ones.
Evaluation holds out one platform at a time and needs at least two per application.
Selection scores show performance relative to the platform's best measured build
(1.0 = best); `error` is mean absolute prediction error.

Use `--app llama.cpp` or `--app gromacs` to select an application, and
`--data path/to/runs.zip` to use other benchmark folders or ZIPs. Both work with
`train`, `suggest`, and `evaluate`; `predict` supports `--app`.

## Ridge Regression models

Ridge regression predicts a target as a weighted sum of input features while
shrinking coefficients to reduce overfitting. Here, Bayesian ridge predicts log
relative performance and estimates uncertainty. The variants add different
interaction features:

| `--model` | Individual options | Option × hardware | Option × option | Option × option × hardware |
| --- | --- | --- | --- | --- |
| `hardware` (default) | Yes | Yes | No | No |
| `hardware-interactions` | Yes | Yes | Yes | No |
| `hardware-three-way` | Yes | Yes | Yes | Yes |

An interaction means that one setting changes another setting's effect:

- **Option × hardware:** flag A helps more on some machines.
- **Option × option:** flags A and B work better (or worse) together.
- **Option × option × hardware:** that extra joint effect differs between machines.

Choose any model from the table using `--model`, for example:

```bash
python -m ml.main train --model hardware-three-way
python -m ml.main predict --model hardware-three-way --system path/to/system.json
python -m ml.main suggest --model hardware-three-way --system path/to/system.json
```

Each model is saved separately. `evaluate` compares all models automatically:
selection scores closer to **1.0** are better; `error*` values closer to **0** are
better. More interactions need more varied measurements—check evaluation results
before choosing a more complex model. Interaction models score all valid build
combinations, which can be expensive with many flags.

## Baseline models

- **`no-hardware`:** Ridge regression using individual build-option effects only;
  hardware is used for build validity, not performance prediction.
- **`no-hardware-interactions`:** Adds option × option effects without hardware
  inputs, so it can learn which flags work well together.
- **`random`:** Chooses a valid configuration uniformly at random without learning
  performance; use `--seed` with `predict` for repeatable choices.
- **Best-on-average (`baseline` in evaluation):** Picks the complete configuration
  with the highest mean relative performance on training platforms among those
  measured and eligible on the held-out platform. Available in evaluation only.

Select a baseline with `--model`, for example `train --model no-hardware`.

## Files

All paths below are inside `ml/`.

| File | Purpose |
| --- | --- |
| `main.py` | CLI commands and model selection. |
| `load.py` | Loads benchmarks and hardware; averages repeated measurements. |
| `load_llama.py` | Parses llama.cpp build options and throughput; checks application-specific validity. |
| `load_gromacs.py` | Parses GROMACS build options and throughput; checks application-specific validity. |
| `models/ridge_regression.py` | Original Bayesian ridge model. |
| `models/ridge_regression_interactions.py` | Extends ridge encoding with pairwise build-option effects. |
| `models/ridge_regression_three_way.py` | Extends pairwise ridge with option-pair × hardware effects. |
| `models/baseline_models.py` | No-hardware, random, and best-on-average baselines. |
| `models/common.py` | Shared candidate validation, selection, suggestions, and save/load. |
| `evaluate.py` | Holds out each platform and compares predicted build choices against measured performance. |
| `validation.py` | Shared option schemas and hardware/toolchain checks, using system metadata. |
| `requirements.txt` | Python dependencies for data handling, modelling, and model storage. |

## Adding a model

1. Create `ml/models/<name>.py` and subclass `BaseModel` from `.common`.
2. Implement `fit(table)`: call `self._fit_options(table)`, train, and return `self`.
3. Implement `predict_performance(hardware, configs)`: return arrays of relative
   performance predictions and uncertainty, both in performance space.
4. Register the class in `MODEL_TYPES` in `ml/main.py`.

The base class handles valid candidates, selection, suggestions, and save/load.

Run model tests with `python -m unittest discover -s ml/tests -v`.
