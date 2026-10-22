#!/usr/bin/env python3
"""Export numeric GPU metrics from an HPCToolkit v4 hpcprof database.

Usage: python3 extract_gpu_metrics.py DATABASE -o gpu-metrics.json
Uses only the Python standard library. Metric names are JSON keys, never values.
"""

import argparse
import json
import math
import mmap
import re
import struct
from pathlib import Path


class DB:
    def __init__(self, path, kind):
        self.file = open(path, "rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:14] != b"HPCTOOLKIT" + kind or self.data[14] != 4:
            raise ValueError(f"{path}: expected HPCToolkit {kind.decode()} v4 database")

    def get(self, fmt, offset):
        return struct.unpack_from("<" + fmt, self.data, offset)

    def string(self, offset):
        if not offset:
            return ""
        end = self.data.find(b"\0", offset)
        if end < 0:
            raise ValueError("unterminated database string")
        return self.data[offset:end].decode("utf-8", errors="replace")

    def section(self, index):
        size, pointer = self.get("QQ", 16 + 16 * index)
        if pointer + size > len(self.data):
            raise ValueError("database section outside file")
        return pointer

    def close(self):
        self.data.close()
        self.file.close()


def metric_ids(meta):
    section = meta.section(2)
    metrics, count, stride, _, summary_stride = meta.get("QIBBB", section)
    scopes, scope_count, scopes_stride = meta.get("QHB", section + 16)
    scope_names = {
        scopes + i * scopes_stride: meta.string(meta.get("Q", scopes + i * scopes_stride)[0])
        for i in range(scope_count)
    }
    summaries = {}
    for i in range(count):
        offset = metrics + i * stride
        name_ptr, _, stats, _, n_stats = meta.get("QQQHH", offset)
        name = meta.string(name_ptr)
        if not name.startswith("G"):
            continue
        for j in range(n_stats):
            stat = stats + j * summary_stride
            scope_ptr, formula, combine, ident = (
                *meta.get("QQB", stat), meta.get("H", stat + 18)[0]
            )
            if scope_names.get(scope_ptr) == "execution" and combine == 0 and meta.string(formula) == "$$":
                summaries[ident] = name
    return summaries


def profile_values(prof, offset, ids):
    """Read only the implicit global context (0), avoiding inclusive CCT duplication."""
    n_values, values, n_contexts, contexts = prof.get("QQI4xQ", offset)
    for i in range(n_contexts):
        context, start = prof.get("IQ", contexts + 12 * i)
        if context != 0:
            continue
        end = prof.get("Q", contexts + 12 * (i + 1) + 4)[0] if i + 1 < n_contexts else n_values
        result = {}
        for j in range(start, end):
            ident, value = prof.get("Hd", values + 10 * j)
            if ident in ids and math.isfinite(value):
                result[ids[ident]] = value
        return result
    return {}


def numeric_name(name):
    # Preserve the HPCToolkit names (including units) as stable machine keys.
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def extract(database):
    meta = DB(database / "meta.db", b"meta")
    prof = DB(database / "profile.db", b"prof")
    try:
        stat_ids = metric_ids(meta)
        section = prof.section(0)
        profiles, count, _ = prof.get("QIB", section)
        if not count:
            raise ValueError("HPCToolkit database has no profiles")
        summary = profile_values(prof, profiles, stat_ids)
        # HPCToolkit omits zero-valued entries; report only actually observed metrics.
        totals = {numeric_name(k): v for k, v in sorted(summary.items())}
        derived = {}
        for numerator, denominator, label in (
            ("GKER (s)", "GKER:COUNT", "kernel_seconds_per_launch"),
            ("GXCOPY (s)", "GXCOPY:COUNT", "copy_seconds_per_operation"),
            ("GKER:THR_VREG_ACUMU", "GKER:COUNT", "kernel_registers_per_launch"),
            ("GKER:THR_SREG_ACUMU", "GKER:COUNT", "kernel_shared_registers_per_launch"),
            ("GKER:STMEM_ACUMU (b)", "GKER:COUNT", "kernel_static_memory_bytes_per_launch"),
            ("GKER:DYMEM_ACUMU (b)", "GKER:COUNT", "kernel_dynamic_memory_bytes_per_launch"),
            ("GKER:LMEM_ACUMU (b)", "GKER:COUNT", "kernel_local_memory_bytes_per_launch"),
            ("GKER:BLK_THR_ACUMU", "GKER:COUNT", "threads_per_block_per_launch"),
            ("GKER:WARP_ACT_ACUMU", "GKER:WARP_AVL_ACUMU", "active_warp_fraction"),
        ):
            if summary.get(denominator) and numerator in summary:
                derived[label] = summary[numerator] / summary[denominator]
        for name, label in (("GKER (s)", "kernel_time_fraction_of_gpu_operation_seconds"),
                            ("GXCOPY (s)", "copy_time_fraction_of_gpu_operation_seconds"),
                            ("GSYNC (s)", "sync_time_fraction_of_gpu_operation_seconds")):
            if summary.get("GPUOP (s)") and name in summary:
                derived[label] = summary[name] / summary["GPUOP (s)"]
        copy_bytes = sum(value for name, value in summary.items()
                         if name.startswith("GXCOPY:") and name.endswith(" (b)"))
        if summary.get("GXCOPY (s)") and copy_bytes:
            derived["copy_bytes_per_copy_second"] = copy_bytes / summary["GXCOPY (s)"]
        if not any(k.startswith("GKER") for k in summary):
            raise ValueError("No GPU kernel metrics recorded; check GPU profiling and database")
        return {**totals, **derived}
    finally:
        prof.close()
        meta.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.database), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
