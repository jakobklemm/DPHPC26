"""Helpers for model-selection.ipynb: choose historical/local pairs for live sampling.

Two data snapshots are compared, both taken from git so that every machine
sees identical inputs:
  - the snapshot the evaluation ZIP was produced from, and
  - the snapshot live sampling will train on (by default HEAD).
Expensive results are cached as CSV files keyed by the resolved commit hash.
"""

import contextlib
import inspect
import io
import subprocess
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SHORT = {
    "AMD Ryzen 7 5800X 8-Core Processor": "5800X",
    "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics": "8845HS",
    "AMD Ryzen 9 3900X 12-Core Processor": "3900X",
    "AMD Ryzen Threadripper 1950X 16-Core Processor + CUDA device": "TR1950X+CUDA",
    "Cortex-X925 + CUDA device": "GB10",
    "Intel(R) Core(TM) i5-1038NG7 CPU @ 2.00GHz": "i5-1038NG7",
}
STEPS = list(range(3, 26))          # steps reported after the 2 start builds
BEST = [f"best@{n}" for n in STEPS]
KEY = ["platform", "pretrained", "model", "seed"]

# On one target machine the hardware features are constant, so these local
# models are numerically identical; they are counted once.
_RIDGE = {"hardware": "ridge", "no-hardware": "ridge",
          "hardware-interactions": "ridge-pairs", "no-hardware-interactions": "ridge-pairs",
          "hardware-three-way": "ridge-pairs"}


def local_family(model):
    name = model.removeprefix("corrected-")
    return _RIDGE.get(name, name.removeprefix("no-hardware-"))


@contextlib.contextmanager
def quiet():
    """Silence output and pin BLAS to one thread.

    Some fits (notably small-mlp's L-BFGS) change with the BLAS thread count,
    which joblib sets from the number of workers; one thread makes results
    independent of `jobs`.
    """
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), threadpool_limits(1):
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------- evaluation ZIP
def _read_zip_csv(zip_path, name, columns):
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(f"evaluation.{name}.csv"))
        with archive.open(member) as handle:
            return pd.read_csv(handle, usecols=columns, keep_default_na=False, na_values=[""])


def zip_pool_best(zip_path):
    """Best measured throughput per platform in the ZIP's replay pools."""
    frame = _read_zip_csv(zip_path, "final", ["platform", "pool_best_performance"])
    return frame.groupby(frame["platform"].map(SHORT).fillna(frame["platform"]))["pool_best_performance"].first()


def load_eval(zip_path):
    """Per-session metrics and best-so-far traces (% of pool best) from the ZIP."""
    timings = _read_zip_csv(zip_path, "timings", KEY + ["step", "best_observed_percent_of_pool_best"])
    quality = _read_zip_csv(zip_path, "quality", KEY + ["step", "recommended_percent_of_pool_best"])
    for frame in (timings, quality):
        frame["platform"] = frame["platform"].map(SHORT).fillna(frame["platform"])
    best = timings.pivot_table(index=KEY, columns="step", values="best_observed_percent_of_pool_best")
    best.columns = [f"best@{n}" for n in best.columns]
    rec = quality.pivot_table(index=KEY, columns="step", values="recommended_percent_of_pool_best")
    sessions = best[BEST].copy()
    sessions["auc"] = sessions[BEST].mean(axis=1)
    sessions["rec@25"] = rec[25]
    sessions = sessions.reset_index()
    sessions["local"] = sessions["model"].map(local_family)
    return sessions


def quality_checks(zip_path):
    """Why final accuracy and timing are not used for ranking.

    already_measured: share of rounds whose recommended build was already
    benchmarked in that session (lower bound: the 2 start builds are not in
    the CSVs). seconds_per_round: model time (predict + select + refit).
    """
    quality = _read_zip_csv(zip_path, "quality", KEY + [
        "step", "recommended_build", "attempted_build",
        "prediction_seconds", "next_sample_seconds", "refit_seconds"])
    quality = quality.sort_values(KEY + ["step"])
    measured = []
    for _, group in quality.groupby(KEY, sort=False):
        seen = set()
        for recommended, attempted in zip(group.recommended_build, group.attempted_build):
            measured.append(recommended in seen)
            seen.add(attempted)
    quality["already_measured"] = measured
    quality["kind"] = np.where(quality.pretrained == "none", "cold", "warm")
    share = quality.pivot_table(index="step", columns="kind", values="already_measured").loc[[5, 10, 15, 25]]
    quality["seconds"] = quality[["prediction_seconds", "next_sample_seconds", "refit_seconds"]].sum(axis=1)
    per_policy = quality.groupby(["kind", "pretrained", "model"]).seconds.mean()
    seconds = per_policy.groupby("kind").describe()[["min", "50%", "max"]]
    seconds["slowest policy"] = per_policy.groupby("kind").idxmax().map(lambda key: f"{key[1]} → {key[2]}")
    return {"already_measured": share.rename_axis(columns="share of recommendations already measured"),
            "seconds_per_round": seconds}


# ---------------------------------------------------------------- data snapshots
def resolve(rev):
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", rev],
                          check=True, capture_output=True, text=True).stdout.strip()


def snapshot(rev, cache):
    """Extract llama/results at a git revision into the cache.

    Returns (commit, data, path). `data` is the git tree hash of llama/results:
    it changes only when the results change, so caches keyed by it survive
    commits that do not touch the data.
    """
    commit, data = resolve(rev), resolve(f"{rev}:llama/results")
    dest = Path(cache) / f"data-{data}"
    if not (dest / "llama" / "results").exists():
        dest.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--prefix=llama/results/", data],
                                 check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(dest, filter="data")
    return commit, data, dest / "llama" / "results"


def load_pools(path):
    from ml.load import load_table
    with quiet():
        return load_table([str(path)], app="llama.cpp", skip_invalid_json=True)


def cached(path, compute, recompute=False):
    path = Path(path)
    if path.exists() and not recompute:
        return pd.read_csv(path)
    frame = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _pool(table, platform):
    from ml.load import app_options
    return app_options(table[table.platform == platform]).reset_index(drop=True)


def _historical(table, name, seed, platform):
    """Historical model fitted on every platform except the target."""
    from ml.load import app_options
    from ml.models import MODEL_TYPES
    cls = MODEL_TYPES[name]
    kwargs = {"random_state": seed} if "random_state" in inspect.signature(cls.__init__).parameters else {}
    return cls(**kwargs).fit(app_options(table[table.platform != platform]))


def is_stochastic(name):
    from ml.models import MODEL_TYPES
    return "random_state" in inspect.signature(MODEL_TYPES[name].__init__).parameters


def _trace(result):
    return {k: 100 * v for k, v in result.items() if k.startswith("best@")}


# ---------------------------------------------------------------- experiments
def _random_job(table, platform, seed):
    from ml.active_sampling import _session
    with quiet():
        result = _session("random", _pool(table, platform), 25, 2, seed)
    return dict(platform=SHORT.get(platform, platform), seed=seed, **_trace(result))


def random_baseline(table, seeds=30, jobs=8):
    """Random search, `seeds` sessions per platform (same rule as the evaluation)."""
    rows = Parallel(n_jobs=jobs)(delayed(_random_job)(table, p, s)
                                 for p in sorted(table.platform.unique()) for s in range(seeds))
    return pd.DataFrame(rows)


def _cold_job(table, name, platform, seed):
    from ml.active_sampling import _session
    with quiet():
        result = _session(name, _pool(table, platform), 25, 2, seed)
    return dict(model=name, local=local_family(name), seed=seed,
                platform=SHORT.get(platform, platform), **_trace(result))


def cold_replay(table, models, seeds=3, jobs=8):
    """Cold-start sessions (no historical model), same seeds and start rule as the evaluation."""
    work = [(m, p, s) for m in models for p in sorted(table.platform.unique()) for s in range(seeds)]
    return pd.DataFrame(Parallel(n_jobs=jobs)(delayed(_cold_job)(table, *w) for w in work))


def _start_job(table, name, seed, platform):
    with quiet():
        model = _historical(table, name, seed, platform)
        pool = _pool(table, platform)
        from ml.load import HW, columns, strip
        from ml.models.common import configs
        hardware = {strip(c): pool[c].iloc[0] for c in columns(pool, HW)}
        mean, _ = model.predict_performance(hardware, configs(pool))
        rho = spearmanr(mean, pool.relative).statistic
    rel = 100 * pool.relative.to_numpy()
    order = np.argsort(-mean)
    return dict(historical=name, seed=seed, platform=SHORT.get(platform, platform), pool=len(pool),
                top1=rel[order[0]], top2=rel[order[:2]].max(), spearman=rho)


def start_picks(table, models, seeds=5, jobs=8):
    """Quality of each historical model's 2 start builds on every held-out platform.

    Stochastic models are refitted with random_state 0..seeds-1, deterministic ones once.
    """
    work = [(m, s, p) for m in models for s in (range(seeds) if is_stochastic(m) else [0])
            for p in sorted(table.platform.unique())]
    return pd.DataFrame(Parallel(n_jobs=jobs)(delayed(_start_job)(table, *w) for w in work))


def _replay_job(table, name, seed, platform, locals_):
    from ml.active_sampling import _session
    rows = []
    with quiet():
        source = _historical(table, name, seed, platform)
        pool = _pool(table, platform)
        for local in locals_:
            # Warm sessions are deterministic given the historical model (session seed 0)
            result = _session(local, pool, 25, 2, 0, source)
            rows.append(dict(historical=name, local=local_family(local), seed=seed,
                             platform=SHORT.get(platform, platform), **_trace(result)))
    return rows


def replay(table, historical, locals_, seeds=5, jobs=8, vary=None):
    """Full 25-step warm-start replays.

    Historical models in `vary` (default: every model with a random_state) are
    refitted with random_state 0..seeds-1; the others once.
    """
    vary = {h for h in historical if is_stochastic(h)} if vary is None else set(vary)
    work = [(h, s, p) for h in historical for s in (range(seeds) if h in vary else [0])
            for p in sorted(table.platform.unique())]
    out = Parallel(n_jobs=jobs)(delayed(_replay_job)(table, *w, locals_) for w in work)
    return pd.DataFrame([row for rows in out for row in rows])


def with_auc(frame):
    frame = frame.copy()
    frame["auc"] = frame[BEST].mean(axis=1)
    return frame
