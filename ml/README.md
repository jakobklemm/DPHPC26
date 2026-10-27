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

Optional: select an application or data source, or train and use a baseline:

```bash
python -m ml.main train --app gromacs --data path/to/runs.zip
python -m ml.load path/to/runs
python -m ml.main train --model no-hardware
python -m ml.main predict --model no-hardware --system path/to/system.json
python -m ml.main train --model random
python -m ml.main predict --model random --system path/to/system.json --seed 42
```

`--app` accepts `llama.cpp` or `gromacs` on all main commands. `--data` accepts
folders or ZIPs for training, suggestions, and evaluation. Baselines are saved
separately; evaluation compares all methods automatically. Validity checks reject
known option, hardware, and toolchain conflicts but do not guarantee compilation.

## Ridge with build-option interactions

| `--model` | Individual options | Option × hardware | Option × option |
| --- | --- | --- | --- |
| `hardware` (default) | Yes | Yes | No |
| `hardware-interactions` | Yes | Yes | Yes |
| `no-hardware` | Yes | No | No |
| `no-hardware-interactions` | Yes | No | Yes |

The interaction models add products of encoded values from different build
options. For flags A and B, the log-performance prediction gains a term
`coefficient_AB * A * B`: an extra effect when both flags are enabled. Numeric
and boolean build settings are treated as categorical choices. Pairs are selected
from training data only; products that never vary are omitted. Pair effects are
shared across machines; no three-way terms are added. Bayesian ridge fitting,
uncertainty, validity checks, and decision reports are reused.

```bash
python -m ml.main train --model hardware-interactions
python -m ml.main train --model no-hardware-interactions
python -m ml.main predict --model hardware-interactions --system path/to/system.json
python -m ml.main suggest --model hardware-interactions --system path/to/system.json
python -m ml.main evaluate
```

Each variant has its own saved model file. Evaluation compares all variants on
the same held-out platforms and measured candidates. `interactions` and
`no_hardware_interactions` are selection scores; `error_interactions` is prediction
error. More pair features can overfit sparse measurements, so compare held-out
scores. The new no-hardware baseline scores all valid combinations, which can
be expensive for many flags; the original baseline retains its additive search.

## Files

All paths below are inside `ml/`.

| File | How it works / provides |
| --- | --- |
| `main.py` | Command-line entry point for training, prediction, suggestions, and evaluation. |
| `load.py` | Reads run folders/ZIPs, extracts hardware features, and merges repeated builds into a table. |
| `load_llama.py` | Parses llama.cpp build options and throughput; checks application-specific validity. |
| `load_gromacs.py` | Parses GROMACS build options and throughput; checks application-specific validity. |
| `models/ridge_regression.py` | Learns hardware/option effects with Bayesian ridge regression; ranks builds and estimates uncertainty. |
| `models/ridge_regression_interactions.py` | Extends ridge encoding with pairwise build-option effects. |
| `models/baseline_models.py` | Provides option-only regression, random selection, and best-on-average configuration selection. |
| `models/common.py` | Shared base class for option domains, valid candidates, selection, suggestions, save/load, and configuration extraction. |
| `models/__init__.py` | Exports the model classes. |
| `evaluate.py` | Holds out each platform and compares predicted build choices against measured performance. |
| `validation.py` | Shared option schemas and hardware/toolchain checks, using system metadata. |
| `requirements.txt` | Python dependencies for data handling, modelling, and model storage. |
| `__init__.py` | Makes `ml` a Python package. |
| `README.md` | Quick overview, commands, and file guide. |

## Adding a model

1. Create `ml/models/<name>.py` and subclass `BaseModel` from `.common`.
2. Implement `fit(table)`: call `self._fit_options(table)`, train, and return `self`.
3. Implement `predict_performance(hardware, configs)`: return arrays of relative
   performance predictions and uncertainty, both in performance space.
4. Register the class in `MODEL_TYPES` in `ml/main.py`.

The base class handles valid candidates, selection, suggestions, and save/load.

Run model tests with `python -m unittest discover -s ml/tests -v`.
