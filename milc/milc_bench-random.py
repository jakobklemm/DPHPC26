#!/usr/bin/env python3
"""MILC version of llama/llama_gpu_bench-random.py (CPU only): same seed, harmonic toggle counts,
resampling of invalid configurations, execution list and resume behaviour."""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from filter_milc_performance_flags import cpu_info, filter_catalog

ROOT = Path(__file__).resolve().parent
SEED = 20261021
DEFAULT_COUNT = 50


def valid(config, rules):
    values = config["cmake"] | config["compiler_flags"]
    return not any(all(values.get(n) == v for n, v in rule.items()) for rule in rules)


def space(options):
    base = {"cmake": {n: v[0] for n, v in options["cmake"].items()},
            "compiler_flags": {n: v[0] for n, v in options["compiler"].items()}}
    catalogs = {"cmake": options["cmake"], "compiler_flags": options["compiler"]}
    candidates = [(k, n) for k, c in catalogs.items() for n, v in c.items() if len(v) > 1]
    return base, catalogs, candidates


def configuration(base, changes):
    config = {"device": "cpu", "compiler": "clang++", "cmake": base["cmake"].copy(),
              "compiler_flags": base["compiler_flags"].copy()}
    for (kind, name), value in changes:
        config[kind][name] = value
    return config


def sweep_configurations(options):
    """Baseline, then every alternative value alone, to show that each flag changes the code. The
    compiler flags are repeated with AVX2, for this CPU and under each LTO mode."""
    base, catalogs, candidates = space(options)
    groups = [[]] + [[((k, n), v)] for k, n in candidates for v in catalogs[k][n][1:]]
    for name, value in (("-march", "-march=x86-64-v3"), ("-march", "-march=native"),
                        ("-flto", "-flto=thin"), ("-flto", "-flto=full")):
        groups += [[(("compiler_flags", name), value), (("compiler_flags", n), v)]
                   for n in options["compiler"] if n != name for v in options["compiler"][n][1:]]
    seen = []
    for group in groups:
        config = configuration(base, group)
        if valid(config, options["invalid_combinations"]) and config not in seen:
            seen.append(config)
            yield {"phase": "sweep", "config": config}


def random_configurations(options, count):
    rng = random.Random(SEED)
    base, catalogs, candidates = space(options)
    counts = range(len(candidates) + 1)
    # Harmonic weights favor small changes without excluding any toggle count.
    weights = [1 / (k + 1) for k in counts]
    for _ in range(count):
        for _attempt in range(10_000):
            toggles = rng.choices(counts, weights=weights, k=1)[0]
            choices = rng.sample(candidates, toggles)
            config = configuration(base, [((k, n), rng.choice(catalogs[k][n][1:])) for k, n in choices])
            if valid(config, options["invalid_combinations"]):
                yield {"phase": "random", "config": config}
                break
        else:
            raise ValueError("Could not sample a valid configuration after 10000 attempts")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    Path(f.name).chmod(0o644)
    Path(f.name).replace(path)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("sweep", "random"))
    p.add_argument("--count", type=positive_int, default=DEFAULT_COUNT, help="random mode only")
    p.add_argument("--flags-file", type=Path, default=ROOT / "flags.json")
    p.add_argument("--build-dir", type=Path, default=Path("/tmp/milc-build"))
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--workers", type=positive_int, default=4)
    p.add_argument("--threads", type=positive_int, default=8)
    p.add_argument("--steps", type=positive_int, default=10)
    p.add_argument("--repetitions", type=positive_int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    results = args.results_dir.resolve()
    architecture, features = cpu_info()
    options = filter_catalog(json.loads(args.flags_file.read_text()), architecture, features)
    plan = list(sweep_configurations(options) if args.mode == "sweep"
                else random_configurations(options, args.count))
    total = len(plan)
    record_file = results / f"{args.mode}-executions.json"
    records = json.loads(record_file.read_text()) if record_file.exists() else []
    if len(records) > total:
        p.error("--count is smaller than the saved execution list")
    for i, record in enumerate(records):
        if record["config"] != plan[i]["config"]:
            p.error(f"Execution {i + 1} differs from this run: {record_file}")
    done = sum(r["status"] == "completed" for r in records)
    print(f"{args.mode}: {total} configurations, {done} completed, {len(records) - done} failed "
          f"(seed {SEED})", flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    settings = {"device": "cpu", "threads": args.threads, "steps": args.steps,
                "repetitions": args.repetitions, "workers": args.workers,
                "inputs": (ROOT / "inputs_bench.template").read_text(),
                "sampling": "harmonic-toggle-count-v1", "seed": SEED, "options": options}
    settings_file = results / f"{args.mode}-settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        p.error(f"Run settings changed: use a different --results-dir or remove {settings_file}")
    save_json(settings_file, settings)
    system_file = results / "host-system-cpu.json"
    if not system_file.exists():
        lscpu = subprocess.check_output(["lscpu"], text=True)
        model = next((l.partition(":")[2].strip() for l in lscpu.splitlines()
                      if l.lower().startswith("model name")), "")
        save_json(system_file, {
            "CPU Info": {"Architecture": architecture, "Model Name": model, "Flags": sorted(features)},
            "Compilers": {"c": {"clang": "installed"}, "cxx": {"clang++": "installed"}},
            "GPU Backends": {}, "Accelerators": {"Accelerators": {}}})
    for index, item in enumerate(plan):
        previous = records[index] if index < len(records) else None
        if previous and previous["status"] == "completed":
            continue
        changed = [f"{n}={v}" for k, c in (("cmake", options["cmake"]), ("compiler_flags", options["compiler"]))
                   for n, v in item["config"][k].items() if v != c[n][0]]
        print(f"\n{'Retrying' if previous else 'Execution'} {index + 1}/{total}: "
              f"{', '.join(changed) or 'baseline'}", flush=True)
        command = [sys.executable, str(ROOT / "milc_bench-profile.py"),
                   "--build-dir", str(args.build_dir), "--results-dir", str(results),
                   "--system-info", str(system_file), "--config", json.dumps(item["config"]),
                   "--workers", str(args.workers), "--threads", str(args.threads),
                   "--steps", str(args.steps), "--repetitions", str(args.repetitions)]
        start = time.monotonic()
        directory = None
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("Run directory: "):
                    directory = Path(line.removeprefix("Run directory: ").strip())
            code = process.wait()
        if directory is None:
            print("No run directory reported; nothing recorded.", file=sys.stderr, flush=True)
            return 1
        record = {"phase": args.mode, "config": item["config"], "results_directory": directory.name,
                  "elapsed_seconds": round(time.monotonic() - start, 3),
                  "status": "failed" if code else "completed"}
        if code:
            record["exit_code"] = code
        if previous:
            record["previous_attempts"] = [*previous.get("previous_attempts", []), {
                k: previous[k] for k in ("results_directory", "elapsed_seconds", "status", "exit_code")
                if k in previous}]
            records[index] = record
        else:
            records.append(record)
        save_json(record_file, records)
    failed = sum(r["status"] == "failed" for r in records)
    print(f"{args.mode} finished: {len(records) - failed}/{total} completed, {failed} failed; "
          f"execution list: {record_file}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
