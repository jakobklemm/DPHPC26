#!/usr/bin/env python3
"""Validate raw HPC flag catalogs without ML dependencies or build execution."""

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED = {"device", "cpu", "compiler", "invalid_combinations"}
ALLOWED = REQUIRED | {"cuda_compiler"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def validate(catalog):
    """Return (errors, warnings); do not infer build or hardware validity."""
    errors, warnings = [], []
    if not isinstance(catalog, dict):
        return ["catalog must be an object"], warnings
    for name in sorted(REQUIRED - catalog.keys()):
        errors.append(f"missing required key: {name}")
    for name in sorted(catalog.keys() - ALLOWED):
        errors.append(f"unknown top-level key: {name}")

    domains = {}

    def option_map(options, location):
        if not isinstance(options, dict):
            errors.append(f"{location} must be an object")
            return set()
        names = set()
        for name, values in options.items():
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{location}: option names must be nonempty strings")
                continue
            names.add(name)
            label = f"{location}.{name}"
            if not isinstance(values, list) or not values:
                errors.append(f"{label} must be a nonempty array")
            elif not all(isinstance(value, str) and value.strip() for value in values):
                errors.append(f"{label}: values must be nonempty strings")
            else:
                if len(set(values)) != len(values):
                    errors.append(f"{label}: duplicate values")
                domains.setdefault(name, set()).update(values)
        return names

    shared = set()
    for section in ("cpu", "compiler", "cuda_compiler"):
        names = option_map(catalog.get(section, {}), section)
        for name in sorted(shared & names):
            errors.append(f"{section}.{name}: option collides with another shared group")
        shared.update(names)

    devices = catalog.get("device", {})
    if not isinstance(devices, dict):
        errors.append("device must be an object of backend option maps")
    else:
        for backend, options in devices.items():
            if not isinstance(backend, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", backend):
                errors.append(f"device: invalid uppercase backend name {backend!r}")
            elif backend == "CPU":
                errors.append("device.CPU: use the top-level cpu map instead")
            names = option_map(options, f"device.{backend}")
            for name in sorted(shared & names):
                errors.append(f"device.{backend}.{name}: option collides with a shared group")

    rules = catalog.get("invalid_combinations", [])
    if not isinstance(rules, list):
        errors.append("invalid_combinations must be an array")
    else:
        for index, rule in enumerate(rules):
            label = f"invalid_combinations[{index}]"
            if not isinstance(rule, dict) or not rule:
                errors.append(f"{label} must be a nonempty object")
                continue
            for name, value in rule.items():
                if name not in domains:
                    errors.append(f"{label}: unknown option {name!r}")
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{label}.{name}: expected a nonempty string")
                elif name in domains and value not in domains[name]:
                    warnings.append(f"{label}: {name}={value!r} is outside the catalog domain; "
                                    "verify the runner's implicit defaults or historical policy")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    try:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    errors, warnings = validate(catalog)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"{args.catalog}: valid catalog structure (not build/ML validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
