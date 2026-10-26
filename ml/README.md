From the repository root, set up once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

Activate `.venv` again in each new terminal.

The default source is the [project Polybox share](https://polybox.ethz.ch/index.php/s/N5f6FHcoFRiCeM3).
The loader discovers ZIPs recursively and reads only the four metadata files per
run using HTTP range requests. Large build artifacts are not downloaded or
extracted. If a server does not support ranges, the archive is downloaded to a
temporary file and removed on close. Network access is required; data is read
afresh on each invocation so new uploads are included automatically.

Each run's `source.yml` identifies its application via `project_name`. Supported
applications are `llama.cpp` and `gromacs`; unknown applications are reported and
skipped. Each application gets its own model and application-specific options
and performance metric.

Inspect the data or train all discovered applications:

```bash
python -m ml.load
python -m ml.main train
```

Models are saved to `models/<application>.joblib`. Predict with all available saved
models, suggest new builds for all discovered applications, or evaluate them:

```bash
python -m ml.main predict --system path/to/system.json
python -m ml.main suggest --system path/to/system.json
python -m ml.main evaluate
```

`predict` uses saved models without accessing Polybox. `suggest` trains fresh
models from current data and ranks untested builds by prediction plus uncertainty.
Benchmark those suggestions separately, upload new run ZIPs, and train again.
The target machine's `system.json` is still required. Evaluation holds out each
platform in turn and requires at least two platforms per application.

Two baselines are defined in `ml/baseline_models.py`, alongside the default
hardware-aware model in `ml/model.py`:

- `no-hardware`: Bayesian ridge regression using build options only. Hardware
  features and hardware-determined options (currently CUDA architecture) are
  excluded from regression. Hardware is used only to filter invalid candidates
  and fill in hardware-determined settings when returning a build. Selection
  uses the additive option effects to search in score order, stopping at the
  first valid build without constructing all combinations.
- `random`: records the option values seen in training without learning from
  performance. It tries configurations in uniformly random order, rejecting
  invalid combinations until a valid build is found. Retries are without
  replacement, so selection raises an error if no valid configuration exists.
  Combinations are sampled lazily rather than constructed in advance.

Train and use either baseline with `--model`:

```bash
python -m ml.main train --model no-hardware
python -m ml.main predict --model no-hardware --system path/to/system.json
python -m ml.main train --model random
python -m ml.main predict --model random --system path/to/system.json --seed 42
```

Baseline files are saved separately as `models/<application>.no-hardware.joblib`
and `models/<application>.random.joblib`. The random baseline returns a build
without a predicted performance. Omit `--seed` for a fresh random selection.

`evaluate` compares the hardware-aware model, both new baselines, and the
original best-on-average baseline automatically:

```bash
python -m ml.main evaluate --random-trials 1000 --seed 42
```

Per platform, `model`, `baseline`, and `no_hardware` report the selected build's measured
performance relative to the best measured build on that platform (higher is
better). `random` and `random_std` report the mean and standard deviation across
the requested number of independent random selections. `error` is the
hardware-aware model's mean absolute prediction error. Evaluation defaults to
1000 random selections with seed 0 for reproducibility.

The `baseline` ranks complete configurations by their arithmetic mean
relative performance on the training platforms and selects the highest-ranked
one that has an eligible measurement on the held-out platform. Hardware-determined
options are excluded from configuration matching. Its reported score is
`selected build performance / best measured performance on the held-out platform`:
1.0 is best, and 0.9 means 90% of the best measured throughput. This is a selection
score, not a prediction error. Unlike `no_hardware`, it looks up complete
configurations rather than learning individual option effects. If no complete
configuration matches training, `baseline` is missing (`NaN`). The CLI's average
is the arithmetic mean across platforms with available scores.

All four methods select from measured configurations that pass the
existing validity rules and use option values seen in the training fold. A
platform with no eligible builds reports `runs=0` and missing scores. The
normalizing best measurement is still the best across all measured builds on
that platform. This evaluates ranking of measured builds; use `predict` and
benchmark its output to evaluate new combinations. Validity checks are the
existing application rules, not a configure/build/run check; llama.cpp currently
has no rejection rules, while GROMACS checks SIMD support and FFT consistency.

Optional `--app` selects one application. `--data` replaces Polybox with local
folders, ZIP files, other public share URLs, or a mixture:

```bash
python -m ml.main train --app gromacs
python -m ml.main train --data path/to/local/runs
python -m ml.main suggest --app llama.cpp --system path/to/system.json --data path/to/runs.zip
python -m ml.load path/to/local/runs
```

Run tests with `python -m unittest discover -s ml/tests`.
