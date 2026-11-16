"""Collect N valid LLM build proposals, evaluate each, and select the best."""

import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from ml.live_llama import sampling_space, write_json
from ml.load import hardware_features, validation_errors
from ml.load_llama import build_argument
from ml.search import CandidateSearch
from ml.validation import compiler_flags, flag_catalog


PROPOSE = """Recommend one fast llama.cpp build for the supplied system and workload.
Return only compact JSON: {"config": {option: value}, "rationale": "brief explanation"}.
Include every flag_space key, using only its
allowed string values. Obey invalid_combinations and the hardware capabilities.
Do not estimate runtime yet. Context is supporting evidence.
The sample number identifies an independent proposal, not an optimization round.
"""

EVALUATE = """Estimate runtime for the supplied validated llama.cpp configuration.
Return only compact JSON: {"estimated_seconds": number, "rationale": "brief explanation"}.
Estimate absolute execution seconds: mean prompt-processing time plus mean
token-generation time, excluding compilation, loading and warmup. Estimates are
not measurements; do not normalize them to 1. Context is supporting evidence.
Do not change the configuration.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--context", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.count > 0, "count must be positive"
    system = json.loads(args.system.read_text())
    application = json.loads(args.application.read_text())
    assert application["name"] == "llama.cpp", "only llama.cpp is supported for now"
    device = application.get("backend", "cpu")
    assert device in {"cpu", "cuda"}, "use an existing CPU/CUDA sampling profile"
    catalog = flag_catalog(application["name"])
    values = {**catalog["cpu"], **catalog["compiler"]}
    if device == "cuda":
        values.update(catalog["device"]["CUDA"])
    values = sampling_space(values, device)
    for language, option in (("c", "CMAKE_C_COMPILER"), ("cxx", "CMAKE_CXX_COMPILER")):
        values[option] = [application["toolchain"][language]]
    hardware = hardware_features(system)
    search = CandidateSearch(values, hardware, application["name"])
    evidence = dict(system=system, application=application, flag_space=values,
                    hardware=hardware, invalid_combinations=catalog["invalid_combinations"],
                    context=[path.read_text() for path in args.context])
    model = os.environ.get("OPENAI_MODEL", "mimo-v2.6")
    url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace, candidates = [], []

    def ask(prompt, data):
        payload = dict(model=model, stream=False, max_tokens=16384, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(data, separators=(",", ":"))}])
        trace.append({"request": payload})
        write_json(args.output.parent / "trace.json", trace)
        start = time.perf_counter()
        with urlopen(Request(url, data=json.dumps(payload).encode(), headers=headers), timeout=600) as response:
            raw = json.load(response)
        trace[-1].update(response=raw, elapsed_seconds=time.perf_counter() - start)
        write_json(args.output.parent / "trace.json", trace)
        choice = raw["choices"][0]
        assert choice["finish_reason"] == "stop", choice["finish_reason"]
        return json.loads(choice["message"]["content"])

    attempt = 0
    while len(candidates) < args.count:
        attempt += 1
        print(f"Proposal {attempt}: {len(candidates)}/{args.count} valid options collected", flush=True)
        candidate = ask(PROPOSE, {**evidence, "sample": attempt})
        config = candidate["config"]
        if not isinstance(config, dict) or set(config) != set(values):
            print("Discarding proposal: config must contain exactly the flag_space keys", flush=True)
            continue
        if not search.accepts(config):
            print(f"Discarding proposal: {validation_errors(config, hardware, application['name']) or 'outside flag catalog'}", flush=True)
            continue
        print(f"Proposal {attempt} valid; estimating runtime", flush=True)
        estimate = ask(EVALUATE, dict(system=system, application=application, config=config, context=evidence["context"]))
        seconds = estimate["estimated_seconds"]
        assert type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, seconds
        candidates.append({**candidate, **estimate, "id": f"c{len(candidates) + 1}"})
        print(f"Option {len(candidates)}/{args.count}: {seconds:g} estimated seconds", flush=True)
    winner = min(candidates, key=lambda candidate: candidate["estimated_seconds"])
    compiler = compiler_flags(application["name"])
    write_json(args.output, dict(model=model, inputs=evidence, candidates=candidates,
        status="unbenchmarked_proposal", recommendation={**winner,
            "cmake_definitions": [build_argument(k, v) for k, v in winner["config"].items() if k not in compiler],
            "compiler_flags": [build_argument(k, v) for k, v in winner["config"].items() if k in compiler]}))
    print(f"Selected {winner['id']}; results: {args.output}", flush=True)
