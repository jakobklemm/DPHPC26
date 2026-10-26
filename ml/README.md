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

Optional `--app` selects one application. `--data` replaces Polybox with local
folders, ZIP files, other public share URLs, or a mixture:

```bash
python -m ml.main train --app gromacs
python -m ml.main train --data path/to/local/runs
python -m ml.main suggest --app llama.cpp --system path/to/system.json --data path/to/runs.zip
python -m ml.load path/to/local/runs
```

Run tests with `python -m unittest discover -s ml/tests`.
