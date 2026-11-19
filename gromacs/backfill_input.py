"""Fill in the "input" block of old GROMACS benchmark.json / benchmark-cpu.json files.

Usage: python3 backfill_input.py <gromacs-script-dir> <results-dir> [--write]
Without --write it only prints what it would add.
The mdrun command comes from build.json (benchmark_command / cpu_benchmark.command),
which is exactly what the script ran; atoms and dt come from measured0/md.log.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

script_dir, results = Path(sys.argv[1]), Path(sys.argv[2])
write = "--write" in sys.argv
sys.path.insert(0, str(script_dir))
spec = importlib.util.spec_from_file_location("bench", script_dir / "gromacs_gpu_bench-profile.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

summary, problems = Counter(), []
for run in sorted(p for p in results.iterdir() if p.is_dir()):
    build_file = run / "build.json"
    if not build_file.is_file():
        continue
    build = json.loads(build_file.read_text())
    targets = [("benchmark.json", "benchmark", build.get("benchmark_command")),
               ("benchmark-cpu.json", "benchmark-cpu",
                (build.get("cpu_benchmark") or {}).get("command"))]
    for name, subdir, command in targets:
        out = run / name
        if not out.is_file():
            continue
        data = json.loads(out.read_text())
        if "input" in data:
            summary[(name, "already has input")] += 1
            continue
        if not command or not (run / subdir / "measured0" / "md.log").is_file():
            problems.append(f"{out}: no command or md.log")
            continue
        info = bench.describe_input(command, run / subdir / "measured0")
        if info["nsteps"] != data.get("nsteps") or None in (info["atoms"], info["tpr_sha256"]):
            problems.append(f"{out}: mismatch/missing {info}")
            continue
        recorded = build.get("tpr_sha256") if name == "benchmark.json" else \
            build["cpu_benchmark"].get("tpr_sha256")
        if recorded and recorded != info["tpr_sha256"]:
            problems.append(f"{out}: input file changed since run ({recorded} vs {info['tpr_sha256']})")
            continue
        # put "input" right after "threads" so the file stays easy to read
        new = {}
        for k, v in data.items():
            new[k] = v
            if k == "threads":
                new["input"] = info
        new.setdefault("input", info)
        summary[(name, info["name"], info["atoms"], info["nsteps"])] += 1
        if write:
            out.write_text(json.dumps(new, indent=2) + "\n")

for k, v in sorted(summary.items(), key=str):
    print(v, *k)
for p in problems:
    print("PROBLEM:", p)
print("written" if write else "dry run (add --write)")
