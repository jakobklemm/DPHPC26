#!/usr/bin/env python3
"""Run perf stat on an application and export CPU counters to numeric JSON.

Usage: python3 collect_cpu_metrics.py -o cpu-metrics.json --benchmark-output bench.json
       --log bench.log -- /path/to/program args...

Run directly on the host alongside the workload.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = (
    "task-clock", "cycles", "instructions", "cache-references", "cache-misses",
    "branches", "branch-misses", "page-faults", "minor-faults", "major-faults",
    "context-switches", "cpu-migrations",
)
OPTIONAL = (
    "L1-dcache-loads", "L1-dcache-load-misses", "LLC-loads", "LLC-load-misses",
    "dTLB-loads", "dTLB-load-misses", "stalled-cycles-frontend",
    "stalled-cycles-backend", "l2_rqsts.references", "l2_rqsts.miss",
    "cycle_activity.stalls_l3_miss", "resource_stalls.sb",
)
# Platform-specific coherence events: first supported alias for each type.
HITM = ("mem_load_l3_hit_retired.xsnp_hitm", "mem_load_l3_hit_retired.xsnp_fwd",
        "ocr.demand_data_rd.l3_hit.snoop_hitm", "offcore_response.demand_data_rd.l3_hitm")


def parse_stat(text):
    """perf stat -x ';' output: value;unit;event;running_ns;running_pct;..."""
    result = {}
    for line in text.splitlines():
        fields = line.strip().split(";")
        if len(fields) < 3 or not fields[2]:
            continue
        raw, unit, event = fields[:3]
        event = re.sub(r":[ukhGH]+$", "", event)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue  # <not supported>, <not counted>, or a diagnostic
        if not math.isfinite(value):
            continue
        key = re.sub(r"[^a-zA-Z0-9]+", "_", event).strip("_").lower()
        if event == "task-clock" and unit == "msec":
            key, value = "task_clock_seconds", value / 1000
        result[key] = value
    return result


def stat(perf, events, command, output, log):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".perf", delete=True) as raw:
        args = [perf, "stat", "--no-big-num", "-x", ";", "-o", raw.name,
                "-e", ",".join(event + ":u" for event in events), "--", *command]
        with output.open("w") as stdout, log.open("w") as stderr:
            process = subprocess.run(args, stdout=stdout, stderr=stderr)
        data = parse_stat(Path(raw.name).read_text())
        if process.returncode:
            raise RuntimeError(f"perf stat failed (exit {process.returncode}); see {log}")
        if not data:
            raise RuntimeError(f"perf did not count any events; see {log}")
        return data


def supported(perf, event):
    # A probe is necessary: `perf list` also advertises PMUs that this user or
    # container cannot access. A successful exit alone can mean <not counted>.
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".perf", delete=True) as raw:
        probe = subprocess.run([perf, "stat", "--no-big-num", "-x", ";", "-o",
                                raw.name, "-e", event + ":u", "--", "true"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        metrics = parse_stat(Path(raw.name).read_text())
        return probe.returncode == 0 and bool(metrics)


def ratios(metrics):
    result = {}

    def divide(name, numerator, denominator, factor=1):
        if numerator in metrics and metrics.get(denominator):
            result[name] = metrics[numerator] * factor / metrics[denominator]

    divide("instructions_per_cycle", "instructions", "cycles")
    divide("cache_miss_fraction", "cache_misses", "cache_references")
    divide("branch_miss_fraction", "branch_misses", "branches")
    divide("l1d_load_miss_fraction", "l1_dcache_load_misses", "l1_dcache_loads")
    divide("llc_load_miss_fraction", "llc_load_misses", "llc_loads")
    divide("dtlb_load_miss_fraction", "dtlb_load_misses", "dtlb_loads")
    divide("cache_misses_per_1000_instructions", "cache_misses", "instructions", 1000)
    divide("l2_misses_per_1000_instructions", "l2_rqsts_miss", "instructions", 1000)
    divide("l2_request_miss_fraction", "l2_rqsts_miss", "l2_rqsts_references")
    divide("hitm_per_1000_instructions", "hitm", "instructions", 1000)
    divide("frontend_stall_cycles_fraction", "stalled_cycles_frontend", "cycles")
    divide("backend_stall_cycles_fraction", "stalled_cycles_backend", "cycles")
    divide("l3_miss_stall_cycles_fraction", "cycle_activity_stalls_l3_miss", "cycles")
    divide("store_buffer_stall_cycles_fraction", "resource_stalls_sb", "cycles")
    divide("context_switches_per_second", "context_switches", "task_clock_seconds")
    return result


def collect(perf, command, output, benchmark, log):
    metrics = stat(perf, BASE, command, benchmark, log)
    if "cycles" not in metrics or "instructions" not in metrics:
        raise RuntimeError(f"perf cannot access CPU hardware counters; see {log}")
    enabled = [event for event in OPTIONAL if supported(perf, event)]
    hitm = next((event for event in HITM if supported(perf, event)), None)
    if hitm:
        enabled.append(hitm)
    # Small batches minimize counter multiplexing, and each is a separate
    # benchmark execution. All counts refer to the benchmark including setup.
    for run, start in enumerate(range(0, len(enabled), 4), 1):
        batch = enabled[start:start + 4]
        more = stat(perf, batch, command,
                    benchmark.with_name(f"cpu-perf-{run}-benchmark.json"),
                    log.with_name(f"cpu-perf-{run}.log"))
        for event in batch:
            key = re.sub(r"[^a-zA-Z0-9]+", "_", event).strip("_").lower()
            if key in more:
                metrics["hitm" if event == hitm else key] = more[key]
    return {**metrics, **ratios(metrics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--perf", default="perf")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the benchmark after --")
    try:
        result = collect(args.perf, command, args.output, args.benchmark_output, args.log)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"CPU profiling failed: {error}\n")


if __name__ == "__main__":
    main()
