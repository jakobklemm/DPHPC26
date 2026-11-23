"""Finish the interrupted llama.cpp test simulation (started 2026-11-19 with ml.main evaluate).

Runs only the sessions (on targets not matching SKIP) that are not complete yet in test-subset/ or test-subset-rest/ and appends
them to test-subset-rest/ (same evaluation loop as ml/evaluate.py:evaluate_sampling).
Safe to interrupt and rerun: a session counts as done once all its timing rows are written
(timings.csv is written last), and half-written sessions are removed from test-subset-rest/ at start.

From the repository root:
    nohup .venv/bin/python evaluation/results/resume_test_subset.py > evaluation/results/llama.cpp/test-subset-rest.log 2>&1 &
"""
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.active_sampling import _session, seeded  # noqa: E402
from ml.evaluation_trace import save_tables  # noqa: E402
from ml.load import app_options, load_table  # noqa: E402
from ml.models import MODEL_TYPES  # noqa: E402

# The original command: ml.main evaluate --app llama.cpp --data llama/results --prompt 512 --generate 128
#   --steps 25 --initial 2 --seeds 3 --jobs 3 --model bayes-pairs no-hardware-gaussian-process sparse-bayes random
#   --pretrained none no-hardware-interactions support-vector small-mlp hardware --output .../test-subset
DONE = ROOT / "evaluation/results/llama.cpp/test-subset"
OUT = ROOT / "evaluation/results/llama.cpp/test-subset-rest"
MODELS = ["bayes-pairs", "no-hardware-gaussian-process", "sparse-bayes", "random"]
HISTORICAL = ["none", "no-hardware-interactions", "support-vector", "small-mlp", "hardware"]
SEEDS, STEPS, INITIAL, JOBS = range(0, 3), 25, 2, 3
SKIP = ("CUDA",)   # target platforms left out: their GPU measurements will be redone (2026-11-19)
KEY = ["platform", "pretrained", "model", "seed"]
TABLES = ["convergence", "improvement", "quality", "final", "sampling", "timings"]


def complete_sessions(folder):
    path = folder / "evaluation.timings.csv"
    if not path.exists():
        return set()
    timings = pd.read_csv(path, keep_default_na=False, na_values=[""])
    counts = timings.groupby(KEY).step.nunique()
    return {tuple(k) for k in counts[counts == STEPS - INITIAL].index}


def repair(folder):
    # Drop a torn last line and every row of sessions whose timings are incomplete.
    for name in TABLES:
        path = folder / f"evaluation.{name}.csv"
        if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            data = path.read_bytes()
            path.write_bytes(data[:data.rfind(b"\n") + 1])
    done = complete_sessions(folder)
    for name in TABLES:
        path = folder / f"evaluation.{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=str)
        keep = [(p, h, m, int(s)) in done for p, h, m, s in frame[KEY].itertuples(index=False)]
        if not all(keep):
            frame[keep].to_csv(path, index=False)
            print(f"removed {len(keep) - sum(keep)} rows of half-written sessions from {path.name}", flush=True)
    return done


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = complete_sessions(DONE) | repair(OUT)
    print(f"{len(done)} sessions already complete", flush=True)
    table = load_table([str(ROOT / "llama/results")], "llama.cpp", skip_invalid_json=True, prompt=512, generate=128)
    remaining = 0
    for app, runs in table.groupby("application"):
        for platform, pool in runs.groupby("platform"):
            if any(word in platform for word in SKIP):
                print(f"skipping target {platform}", flush=True)
                continue
            pool = app_options(pool).reset_index(drop=True)
            for source_name in HISTORICAL:
                local = [name for name in MODELS if source_name == "none" or name != "random"]
                todo = [(name, seed) for name in local for seed in SEEDS
                        if (platform, source_name, name if source_name == "none" else f"corrected-{name}", seed) not in done]
                if not todo:
                    continue
                remaining += len(todo)
                sources = dict.fromkeys(SEEDS)
                if source_name != "none":
                    train = app_options(runs[runs.platform != platform])
                    fits = {}
                    for seed in sorted({seed for _, seed in todo}):
                        model_type = seeded(source_name, seed)
                        key = seed if model_type is not MODEL_TYPES[source_name] else None
                        if key not in fits:
                            fits[key] = model_type().fit(train)
                        sources[seed] = fits[key]
                print(f"{app} / {platform}: historical={source_name}, {len(todo)} sessions", flush=True)
                results = Parallel(n_jobs=JOBS, return_as="generator_unordered")(
                    delayed(_session)(name, pool, STEPS, INITIAL, seed, sources[seed],
                                      evaluation=True, random_state=seed)
                    for name, seed in todo)
                for result in results:
                    tables = result.pop("_evaluation")
                    identity = dict(application=app, platform=platform, pretrained=source_name,
                                    model=result["model"], seed=result["seed"],
                                    workload=pool["workload"].iloc[0] if "workload" in pool else "")
                    save_tables(OUT / "evaluation", tables, identity, append=True)
    print(f"finished: {remaining} sessions run, results in {OUT}", flush=True)


if __name__ == "__main__":
    main()
