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

## Files

All paths below are inside `ml/`.

| File | How it works / provides |
| --- | --- |
| `main.py` | Command-line entry point for training, prediction, suggestions, and evaluation. |
| `load.py` | Reads run folders/ZIPs, extracts hardware features, and merges repeated builds into a table. |
| `load_llama.py` | Parses llama.cpp build options and throughput; checks application-specific validity. |
| `load_gromacs.py` | Parses GROMACS build options and throughput; checks application-specific validity. |
| `model.py` | Learns hardware/option effects with Bayesian ridge regression; ranks builds and estimates uncertainty. |
| `baseline_models.py` | Provides option-only regression, random selection, and best-on-average configuration selection. |
| `evaluate.py` | Holds out each platform and compares predicted build choices against measured performance. |
| `validation.py` | Shared option schemas and hardware/toolchain checks, using system metadata. |
| `requirements.txt` | Python dependencies for data handling, modelling, and model storage. |
| `__init__.py` | Makes `ml` a Python package. |
| `README.md` | Quick overview, commands, and file guide. |
