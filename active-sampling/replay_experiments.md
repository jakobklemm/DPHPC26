# Reproduce the llama.cpp policy sweep

`run_all_policies.py` is a portable version of the temporary
`/tmp/dphpc_fullgrid.py` used for the experiment. The original experiment combined
several batches and reused earlier sessions. This version can run all 421 policies
from scratch: 20 cold models, 400 ordered historical/local combinations, and random.
It uses the repository's `ml.active_sampling._session` and estimator defaults.

Run from the repository root with the project's Python environment:

> **Start rule changed (2026-11-10):** with a pretrained model, the `--initial` start builds are now all its top picks (before: half top picks, half random), and the default is 2. The commands below with `--initial 8` reproduce the old settings' size, not the old results.

```bash
.venv/bin/python active-sampling/run_all_policies.py \
  --data llama/results results-500-seed20261029.zip \
  --seeds 3 --steps 25 --initial 8 --jobs 6 --compact-bayes
```

The default output is `active-sampling/llama-all-policies-replay.csv`. Each completed
session is appended immediately. Repeating the command resumes missing sessions.
`--seeds` repeats cold-start policies only. Each historical/local policy runs once,
with seed 0, because its choices are deterministic. The updated metadata schema
requires a fresh output path for results generated before this change.
A metadata sidecar checks the loaded data, steps, initial sample count, start rule, and numerical
implementation before resuming. Use a new `--output` for different experiments.
Run only one writer against a given output file at a time.

Explicit ZIP inputs are copied into temporary sanitized archives, excluding macOS
resource-fork metadata and malformed benchmark JSON. Exclusions are printed. This
reproduces the treatment of the one malformed record in the additional Vulkan ZIP.
The original archives remain the inputs for subsequent runs.

Compact Bayesian fitting is now integrated into the ridge models and `bayes-pairs`
for all commands. It fits in the reduced training row space and preserves prior
uncertainty outside that space, without a feature-by-feature covariance matrix.
`--compact-bayes` remains accepted for compatibility but is no longer needed.
The old helper no longer patches sklearn source code. Use a fresh output path
when switching from results produced by the earlier implementation.

For each target, the historical model is fitted only on the other targets. Its
predictions are cached for the fixed replay pool. Every session selects builds,
reveals their recorded performance, and refits the local model using the same UCB
policy as the repository. The CSV reports `best@n` for steps after startup.
`best@25` is a fraction of the target pool's recorded best
throughput; multiply it by 100 for a percentage. Average within each target and then
across targets to give each target equal weight.

## Extend a cold-start policy to ten seeds

Use the same output and data with a specific cold-start model. Existing seeds
are reused and seeds 3–9 are added:

```bash
.venv/bin/python active-sampling/run_all_policies.py \
  --data llama/results results-500-seed20261029.zip \
  --historical none --model gaussian-process \
  --seeds 10 --steps 25 --initial 8 --jobs 6 --compact-bayes
```

The following describes the earlier experiment, before redundant warm-start seed repeats were removed.
For the reported experiment, the top five distinct pairings at 15 builds and the top
five at 25 builds were selected from the three-seed screen. Local hardware/no-hardware
variants were counted once during selection. Their union contained nine warm
policies. Together with earlier ten-seed runs, the final dataset contains 421
three-seed policies and 39 ten-seed policies, totaling 4,608 distinct sessions.

For a small execution check, use `--historical none --model random --seeds 1
--steps 9 --initial 8 --output /tmp/llama-replay-smoke.csv`.

## Workload filtering

Replay now selects pp512/tg128 before aggregation by default. Use `--prompt` and
`--generate` to evaluate a different workload separately. Included/excluded raw
run counts are printed per target. Workload selection is recorded in resume
metadata; use a fresh output path for results from before this change.
