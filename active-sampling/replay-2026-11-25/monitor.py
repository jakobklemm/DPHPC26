#!/usr/bin/env python3
"""Progress updates and crash-safe backups for a run_all_policies.py run.

Every --every minutes: print progress (sessions done, per platform, rate, ETA),
check that the main CSV ends with a complete row, and copy the CSV, its sidecar
CSVs and the metadata file to backup/ (atomically, then fsync), so that a power
loss can cost at most the sessions since the last backup.

Usage (from the repository root):
  .venv/bin/python active-sampling/replay-2026-11-25/monitor.py \
      active-sampling/replay-2026-11-25/llama-all-policies.csv --every 10

  --repair   drop an incomplete last row of the main CSV (after a crash), then exit
"""
import argparse
import csv
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

SESSIONS_PER_TARGET = 21 * 3 + 20 * 20   # cold: 21 models x 3 seeds; warm: 20 historical x 20 local
TARGETS = 10


def rows(path):
    """Return (header, complete rows, whether the last line is incomplete)."""
    text = path.read_text(errors="replace")
    if not text:
        return None, [], False
    lines = list(csv.reader(text.splitlines()))
    header, body = lines[0], lines[1:]
    broken = bool(body) and (len(body[-1]) != len(header) or not text.endswith("\n"))
    return header, (body[:-1] if broken else body), broken


def repair(path):
    header, body, broken = rows(path)
    if not broken:
        print(f"{path}: last row is complete, nothing to do")
        return
    shutil.copy2(path, path.with_suffix(".csv.before-repair"))
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(body)
    print(f"{path}: dropped the incomplete last row (copy: {path.with_suffix('.csv.before-repair')})")


def backup(files, target):
    target.mkdir(exist_ok=True)
    for source in files:
        temporary = target / (source.name + ".tmp")
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target / source.name)
    descriptor = os.open(target, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)


def running():
    """True while a Python process runs run_all_policies.py (not a shell that merely mentions it)."""
    for proc in Path("/proc").iterdir():
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        if (len(argv) > 1 and b"python" in Path(argv[0].decode(errors="replace")).name.encode()
                and argv[1].endswith(b"run_all_policies.py")):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--every", type=float, default=10, help="minutes between updates (default 10)")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    if args.repair:
        repair(args.csv)
        return
    total = SESSIONS_PER_TARGET * TARGETS
    log = args.csv.parent / "run.log"
    history = []   # (time, sessions done)
    while True:
        now = datetime.now()
        stamp = now.strftime("%H:%M")
        if not args.csv.exists():
            print(f"[{stamp}] no CSV yet (still loading data / fitting the first models)", flush=True)
        else:
            header, body, broken = rows(args.csv)
            done = len(body)
            history.append((now, done))
            history[:] = [h for h in history if now - h[0] <= timedelta(hours=2)]
            rate = ""
            if len(history) > 1 and history[-1][1] > history[0][1]:
                hours = (history[-1][0] - history[0][0]).total_seconds() / 3600
                per_hour = (history[-1][1] - history[0][1]) / hours
                eta = now + timedelta(hours=(total - done) / per_hour)
                rate = f", {per_hour:.0f}/h, ETA ~{eta:%a %H:%M} (estimate)"
            per_platform = Counter(row[header.index("platform")] for row in body) if body else Counter()
            print(f"[{stamp}] {done}/{total} sessions ({done / total:.1%}){rate}"
                  + ("  WARNING: incomplete last row" if broken else ""), flush=True)
            for platform, count in sorted(per_platform.items()):
                print(f"         {platform[:55]:55} {count:4}/{SESSIONS_PER_TARGET}", flush=True)
            sidecars = sorted(args.csv.parent.glob(args.csv.stem + ".*.csv"))
            extra = [p for p in (args.csv.with_suffix(".metadata.json"), args.csv.with_suffix(".run-info.json"))
                     if p.exists()]
            backup([args.csv, *extra, *sidecars], args.csv.parent / "backup")
            print(f"         backup written ({1 + len(extra) + len(sidecars)} files)", flush=True)
        if log.exists():
            last = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
            if last:
                print(f"         log: {last[-1][:110]}", flush=True)
        if not running():
            print(f"[{stamp}] run_all_policies.py is not running (finished or stopped); last backup above.", flush=True)
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
