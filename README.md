# DPHPC

```sh
docker pull ghcr.io/jakobklemm/dphpc26:cpu

# optional?
docker tag ghcr.io/jakobklemm/dphpc26:cpu dphpc26:cpu

mkdir -p ~/dphpc-data
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu setup --skip-train

docker run --rm -v ~/dphpc-data:/data dphpc26:cpu sweep \
  --workers 4 --prompt 32 --generate 32 --repetitions 1

docker run --rm -v ~/dphpc-data:/data dphpc26:cpu random \
  --count 50
```

## Train & active sampling

```sh
mkdir -p ~/dphpc-data/llama ~/dphpc-data/gromacs
cp llama/flags.json ~/dphpc-data/llama/
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu train --model hardware
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu active-sampling --steps 10
```

## Evaluate active sampling

1. **Setup** (repository root)

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r ml/requirements.txt
   .venv/bin/python -m pip install -r evaluation/requirements-notebook.txt   # optional: analysis notebook
   ```

2. **Smoke test** (optional)

   ```bash
   .venv/bin/python -m ml.main evaluate --app llama.cpp --data llama/results \
     --model hardware gaussian-process random --pretrained none hardware \
     --prompt 512 --generate 128 --steps 3 --seeds 1 --jobs 2 --output evaluation-smoke
   ```

3. **Full simulation** (all model pairs, cold starts, random baseline)

   ```bash
   .venv/bin/python -m ml.main evaluate --app llama.cpp --data llama/results \
     --prompt 512 --generate 128 --steps 25 --initial 2 --seeds 3 --jobs 2 --output evaluation-results-3seeds-25steps
   ```

   The loader selects pp512/tg128 before averaging or normalization and prints
   included/excluded raw-run counts for each target. This excludes the Intel pp32/tg32
   runs and the CPU Threadripper pp128/tg1024 runs. Use matching workloads and model
   weights for comparisons; other workloads require separate evaluation runs.

   Fresh output directory per run. [Resumable replay](active-sampling/replay_experiments.md).

4. **Results** (every step after startup, starting at step 3 with `--initial 2`)

   - `evaluation.timings.csv` — `best_observed_percent_of_pool_best` vs. random per target.
   - `evaluation.quality.csv` — `recommended_percent_of_pool_best`.
   - Cross-hardware comparisons — match model weights and workloads. [CSV definitions](ml/README.md#metrics).

   [Evaluation analysis notebook](evaluation/evaluation-analysis.ipynb) — top-five tables
   and plots for prediction accuracy, improvement, convergence and timing.
   Install `evaluation/requirements-notebook.txt` (step 1), open the notebook
   and select the `.venv` kernel (VS Code: **Select Kernel** →
   **Python Environments** → `.venv`). It reads
   `evaluation-results-3seeds-25steps.zip` next to it; set `RESULTS` to another ZIP, or to a
   results folder to follow a running evaluation. Set `HARDWARE` to an exact
   platform name, or leave it as `None` for every observed platform and a
   shared-coverage average.

5. **Live evaluation** (optional, inside the [prepared container](ml/README.md#1-run-active-sampling-currently-only-for-llama))

   ```bash
   python -m ml.main evaluate --mode live --model gaussian-process --pretrained none \
     --data /data/llama/results --workdir /data --steps 25 --output /data/live-evaluation
   ```

   Warm start — replace `none` with a historical model. Results — `active-*` subdirectory.
