#!/usr/bin/env python3
"""Replay every cold and historical/local policy on llama.cpp benchmark pools.

Portable version of the temporary dphpc_fullgrid.py experiment runner.
Run from the repository root. See replay_experiments.md for commands.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ml.active_sampling import _session
from ml.load import HW, app_options, columns, load_table, strip
from ml.models import MODEL_TYPES
from ml.models.common import configs


class FrozenPoolPredictions:
    """Reuse fixed historical predictions without refitting during a session."""

    def __init__(self, source, pool):
        builds = configs(pool)
        hardware = {strip(c): pool[c].iloc[0] for c in columns(pool, HW)}
        means, spreads = source.predict_performance(hardware, builds)
        self.options = tuple(builds[0])
        self.values = {
            self.key(build): (mean, spread)
            for build, mean, spread in zip(builds, means, spreads)
        }

    def key(self, build):
        return tuple(build[option] for option in self.options)

    def predict_performance(self, hardware, builds):
        return np.asarray([self.values[self.key(build)] for build in builds]).T


def run_case(local, pool, source, source_name, steps, initial, seed, compact):
    if compact:
        from compact_bayes import activate
        activate()
    result = _session(local, pool, steps, initial, seed, source)
    return dict(result, platform=pool.platform.iloc[0],
                pretrained=source_name, application='llama.cpp')


def sanitized_paths(paths, temporary):
    """Exclude macOS ZIP metadata and malformed benchmark JSON, as in the run."""
    cleaned = []
    for index, name in enumerate(paths):
        path = Path(name)
        if path.suffix.lower() != '.zip':
            cleaned.append(path)
            continue
        target = Path(temporary) / f'{index}.zip'
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(target, 'w') as output:
            for entry in source.infolist():
                if entry.filename.startswith('__MACOSX/'):
                    continue
                data = source.read(entry)
                if entry.filename.endswith('/benchmark.json'):
                    try:
                        json.loads(data)
                    except (ValueError, UnicodeDecodeError):
                        print(f'Excluded malformed benchmark: {path}/{entry.filename}', flush=True)
                        continue
                output.writestr(entry, data)
        cleaned.append(target)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', default=[str(ROOT / 'llama/results')])
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'active-sampling/llama-all-policies-replay.csv')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--initial', type=int, default=8)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--model', nargs='+', choices=MODEL_TYPES,
                        help='Restrict local models for a small run; default: every model.')
    parser.add_argument('--historical', nargs='+',
                        choices=['none'] + [name for name in MODEL_TYPES if name != 'random'],
                        help='Restrict historical models; default: none plus every learned model.')
    parser.add_argument('--compact-bayes', action='store_true',
                        help='Use the experiment’s verified reduced-covariance Bayesian ridge fit.')
    args = parser.parse_args()
    if args.seeds < 1 or not 2 <= args.initial <= args.steps or args.jobs == 0:
        parser.error('require seeds >= 1, 2 <= initial <= steps, and jobs != 0')
    if args.compact_bayes:
        from compact_bayes import activate
        activate()
    with tempfile.TemporaryDirectory(prefix='llama-replay-') as temporary:
        table = load_table(sanitized_paths(args.data, temporary), 'llama.cpp')
    pools = [(platform, app_options(pool).reset_index(drop=True))
             for platform, pool in table.groupby('platform')]
    if any(len(pool) <= args.steps for _, pool in pools):
        parser.error('every target pool must contain more builds than --steps')
    local_models = args.model or list(MODEL_TYPES)
    sources = args.historical or ['none'] + [m for m in MODEL_TYPES if m != 'random']
    if any(s != 'none' for s in sources) and len(pools) < 2:
        parser.error('historical fitting requires at least two target platforms')

    # A sidecar prevents resuming an output with different data or sampling rules.
    canonical = table.reindex(columns=sorted(table.columns)).sort_values(
        ['application', 'platform'] + sorted(c for c in table if c.startswith('cfg.')))
    metadata = {'budget': args.steps,  # key kept so existing outputs still resume
                'initial': args.initial,
                'compact_bayes': args.compact_bayes,
                'data_sha256': hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()}
    manifest = args.output.with_suffix('.metadata.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            parser.error('output metadata differs; use a new --output path')
    elif args.output.exists():
        parser.error('existing output has no metadata sidecar; use a new --output path')
    else:
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    seen = set()
    if args.output.exists():
        for row in pd.read_csv(args.output, keep_default_na=False).itertuples():
            seen.add((row.platform, row.pretrained,
                      row.model.removeprefix('corrected-'), row.seed))
    policies = [(source, local) for source in sources for local in local_models
                if source == 'none' or local != 'random']
    print(f'{len(policies)} policies, {len(pools)} targets, {args.seeds} seeds', flush=True)
    fields = ['model', 'seed', *[f'best@{n}' for n in range(1, args.steps + 1)],
              'model_pick', 'rank_correlation', 'seconds', 'platform', 'pretrained', 'application']
    with Parallel(n_jobs=args.jobs, return_as='generator_unordered', batch_size=1) as parallel:
        for platform, pool in pools:
            for source_name in sources:
                pending = [(local, seed) for source, local in policies if source == source_name
                           for seed in range(args.seeds)
                           if (platform, source_name, local, seed) not in seen]
                if not pending:
                    continue
                prior = None
                if source_name != 'none':
                    training = app_options(table[table.platform != platform])
                    prior = FrozenPoolPredictions(MODEL_TYPES[source_name]().fit(training), pool)
                print(f'{platform}: historical={source_name}, {len(pending)} sessions', flush=True)
                results = parallel(delayed(run_case)(local, pool, prior, source_name,
                                   args.steps, args.initial, seed, args.compact_bayes)
                                   for local, seed in pending)
                for result in results:
                    header = not args.output.exists()
                    with args.output.open('a', newline='') as output:
                        writer = csv.DictWriter(output, fieldnames=fields)
                        if header:
                            writer.writeheader()
                        writer.writerow(result)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
