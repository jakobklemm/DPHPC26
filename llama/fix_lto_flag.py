#!/usr/bin/env python3
"""Relabel runs as GGML_LTO=OFF where LTO silently failed.

Before commit 82240b6 the containers had no llvm-ar: CMake printed "IPO is not supported"
and built without LTO, but the run was still recorded as GGML_LTO=ON. This script finds
those runs by their build.log and sets the flag to OFF in build.json, *-executions.json
and best.json. configure_command is kept, as it is the command that was run.

Usage: python3 llama/fix_lto_flag.py <results folder>... [--dry-run]
"""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def lto_off(config):
    """Set GGML_LTO from ON to OFF in a config dict, return whether it changed."""
    if isinstance(config, dict) and config.get("GGML_LTO") == "ON":
        config["GGML_LTO"] = "OFF"
        return True
    return False


def fix_folder(root, dry_run):
    # 1. build.json of every run whose build.log shows the failed LTO check
    fixed = {}  # run dir name -> performance (used to match best.json)
    for log in sorted(root.rglob("build.log")):
        build_path = log.with_name("build.json")
        if "IPO is not supported" not in log.read_text(errors="replace") or not build_path.exists():
            continue
        build = load(build_path)
        # current runs keep the flags in "cmake", older active-learning runs in "config"
        if not (lto_off(build.get("cmake")) | lto_off(build.get("config"))):
            continue
        build["cmake_definitions_in_order"] = [
            "-DGGML_LTO=OFF" if d == "-DGGML_LTO=ON" else d
            for d in build.get("cmake_definitions_in_order", [])]
        build.setdefault("corrections", []).append("GGML_LTO ON -> OFF: CMake reported 'IPO is not supported'")
        fixed[log.parent.name] = build.get("performance")
        print(f"{root.name}/{log.parent.relative_to(root)}")
        if not dry_run:
            save(build_path, build)

    # 2. summary files that repeat the configs of these runs
    for path in root.rglob("*-executions.json"):
        executions = load(path)
        changed = [lto_off(e.get("config", {}).get("cmake"))
                   for e in executions if e.get("results_directory") in fixed]
        if any(changed) and not dry_run:
            save(path, executions)
    for path in root.rglob("best.json"):
        best = load(path)
        if best.get("performance") in set(fixed.values()) - {None} and lto_off(best.get("config")) and not dry_run:
            save(path, best)

    print(f"{root}: {len(fixed)} run(s) {'to fix' if dry_run else 'fixed'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel runs as GGML_LTO=OFF where LTO silently failed.")
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="only list the affected runs")
    args = parser.parse_args()
    for folder in args.folders:
        fix_folder(folder.resolve(), args.dry_run)
