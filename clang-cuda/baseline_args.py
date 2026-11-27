#!/usr/bin/env python3
"""Print gromacs_gpu_bench-profile.py arguments for the sweep baseline of a flag catalog on this host.

Usage: baseline_args.py <flags-file> <cuda-arch> [<option> <value> ...]
  each pair overrides one option, e.g. cuda:-finline-functions -fno-inline-functions
Output: one argument per line (--cmake-arg/--compiler-flag/--cuda-flag), for `mapfile`.
"""
import importlib.util, json, sys
from pathlib import Path

G = Path(__file__).resolve().parent.parent / "gromacs"
sys.path.insert(0, str(G))
spec = importlib.util.spec_from_file_location("runner", G / "gromacs_gpu_bench-random.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
import filter_gromacs_performance_flags as f

flags_file, arch, overrides = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
architecture, features = f.cpu_info()
options = f.filter_catalog(json.loads(flags_file.read_text()), architecture, features, "cuda")
config = runner.execution_plan(options, "cuda", arch, "sweep")[0]["config"]   # sweep[0] = baseline
if len(overrides) % 2:
    sys.exit("overrides must be <option> <value> pairs")
for name, value in zip(overrides[::2], overrides[1::2]):
    for part in ("cmake", "compiler_flags", "cuda_flags"):
        if name in config.get(part, {}):
            config[part][name] = value
            break
    else:
        sys.exit(f"unknown option {name!r}")
for name, value in config["cmake"].items():
    print("--cmake-arg"); print(f"{name}={value}")
for value in config["compiler_flags"].values():
    print(f"--compiler-flag={value}")
for value in config.get("cuda_flags", {}).values():
    print(f"--cuda-flag={value}")
