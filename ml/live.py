"""Live active sampling on this machine (ml.main active-sampling), for any application.

run_session repeats: LiveActiveSampler proposes a build, the application's
benchmark() builds and measures it, the sampler refits. Each build gets a
results folder that later loads as training data.

What is application specific lives in one module per application
(live_llama.py, live_gromacs.py), with:
    APPLICATION, UNIT, RESULTS      name, performance unit, results folder in the work folder
    prepare(work, device, **options) -> context   check the work folder once; context is passed to benchmark
    workload(context)               workload label of the session, as the loader writes it
    sampling_space(values, device, hardware)      flag space for this session
    benchmark(config, context, build, output, system, device, *, timings) -> performance
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .active_sampling import LiveActiveSampler
from .evaluation_trace import EvaluationTrace, save_tables, validate_budget
from .load import CFG, app_options, columns, describe, hardware_features, platform_name, strip


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def detect_device(hardware):
    """CUDA needs the toolkit ("cuda") and a visible NVIDIA GPU; the CUDA image always has the toolkit."""
    gpus = hardware.get("n_gpus")
    has_gpu = hardware.get("gpu_vendor") == "nvidia" and isinstance(gpus, int) and gpus > 0
    return "cuda" if hardware.get("cuda") is True and has_gpu else "cpu"


def run_session(app, model_type, system, table, *, pretrain, steps, initial, seed, device=None,
                workdir, output=None, values=None, settings=None, evaluation=False, options=None):
    """Benchmark `steps` builds of `app` (a live module) proposed by model_type (None: random).

    table: historical runs for the flag space; pretrain(platform): model fitted on
    the other machines, or None. values replaces the flag space from table.
    options: application-specific benchmark settings for app.prepare.
    Returns the results folder.
    """
    validate_budget(steps, initial)
    work = Path(workdir).expanduser().resolve()
    platform, hardware = platform_name(system), hardware_features(system)
    device = device or detect_device(hardware)
    context = app.prepare(work, device, **(options or {}))
    application, unit = app.APPLICATION, app.UNIT

    # Flag space from the benchmarks (or values), backend fixed to the device
    if values is None:
        runs = app_options(table)
        values = {strip(c): sorted(runs[c].unique()) for c in columns(runs, CFG)}
    values = app.sampling_space(values, device, hardware)
    pretrained = pretrain(platform)

    # One results folder per session
    session = "active-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output or work / app.RESULTS / "results").expanduser().resolve() / session
    output.mkdir(parents=True)
    build = work / "build" / session
    write_json(output / "session.json", {**(settings or {}), "application": application,
                                         "platform": platform, "device": device,
                                         "workload": app.workload(context), "space": values})
    settings = settings or {}
    model = settings.get("model") or ("random" if model_type is None else model_type.__name__)
    source = settings.get("pretrained") or (type(pretrained).__name__ if pretrained else "none")
    print(f"== {application} on {platform} ({device}): {steps} steps, model {model}, "
          f"pretrained {source}\n   results: {output}", flush=True)

    trace = EvaluationTrace(initial, include_startup=not evaluation)
    identity = dict(application=application, platform=platform, pretrained=source, model=model, seed=seed,
                    workload=app.workload(context))

    session_start = time.perf_counter()
    sampler = LiveActiveSampler(model_type, values, hardware, application,
                                initial=initial, seed=seed, pretrained=pretrained)
    trace.initialization_seconds = time.perf_counter() - session_start

    def evaluate_round(step, ask_seconds, refit_seconds, status, benchmark_seconds, attempt_timings,
                       attempted_build, attempted_performance=None):
        prediction_seconds = 0.0
        best_observed = max((value for _, value in sampler.measured.values()), default=None)
        successful_before = len(sampler.measured) - (status == 'completed')
        if len(sampler.measured) >= initial and (not evaluation or successful_before >= initial):
            tick = time.perf_counter()
            # Fixed evaluation RNG, isolated from active candidate generation.
            state = sampler.search.rng.getstate()
            try:
                sampler.search.rng.seed(seed)
                if sampler.model is not None:
                    config, score = sampler.predict()
                    first = trace.picks[0]['config'] if trace.picks else config
                    first_scores, _ = sampler.model.predict_performance(hardware, [first])
                    first_score = float(first_scores[0])
                else:
                    config, score, first_score = sampler.recommend(), None, None
            finally:
                sampler.search.rng.setstate(state)
            prediction_seconds = time.perf_counter() - tick
            measured = sampler.measured.get(sampler.search.key(config))
            predicted_performance = None if score is None else score * getattr(
                sampler.model, 'performance_scale', best_observed)
            trace.record(step, len(sampler.measured), config, score, first_score,
                         actual=None if measured is None else measured[1], best_observed=best_observed,
                         predicted_performance=predicted_performance, unit=unit)
            if predicted_performance is not None:
                print(f"   prediction round {trace.picks[-1]['round']}: "
                      f"{predicted_performance:.3f} {unit} estimated; "
                      f"prediction took {prediction_seconds:.6f} s", flush=True)
        trace.timing(step, len(sampler.measured), ask_seconds, refit_seconds, prediction_seconds, status,
                     benchmark_seconds=benchmark_seconds, elapsed_seconds=time.perf_counter() - session_start,
                     attempted_build=attempted_build, attempted_performance=attempted_performance,
                     best_observed=best_observed, unit=unit, **attempt_timings)
        # Checkpoint after each attempt, including failures.
        save_tables(output / 'evaluation', trace.tables(), identity)

    def final_pick():
        """The model's final prediction (as reported per round), or None if it was already attempted."""
        state = sampler.search.rng.getstate()
        try:
            sampler.search.rng.seed(seed)
            config, _ = sampler.predict()
        finally:
            sampler.search.rng.setstate(state)
        key = sampler.search.key(config)
        if key in sampler.measured or key in sampler.failed or not sampler.search.accepts(config):
            return None
        return config.copy()

    best_score = 0.0
    for step in range(1, steps + 1):
        tick = time.perf_counter()
        # Last build of the budget: the model's final prediction, so the result is a measured build
        config = final_pick() if step == steps and sampler.model is not None else None
        if config is None:
            config = sampler.ask()
        else:
            sampler.source = "final model prediction"
        ask_seconds = time.perf_counter() - tick
        print(f"\n-- build {step}/{steps} ({sampler.source}): {describe(application, config)}", flush=True)
        step_output = output / f"{step:04d}"
        step_output.mkdir()
        benchmark_start = time.perf_counter()
        attempt_timings = {}
        try:
            score = app.benchmark(config, context, build, step_output, system, device, timings=attempt_timings)
        except RuntimeError as error:
            # Failed builds count as a step and are not proposed again
            print(f"   failed: {error}", flush=True)
            sampler.fail(config)
            evaluate_round(step, ask_seconds, 0.0, 'failed', time.perf_counter() - benchmark_start, attempt_timings, config)
            continue
        benchmark_seconds = time.perf_counter() - benchmark_start
        tick = time.perf_counter()
        sampler.tell(config, score)
        refit_seconds = time.perf_counter() - tick if sampler.model is not None else 0.0
        evaluate_round(step, ask_seconds, refit_seconds, 'completed', benchmark_seconds,
                       attempt_timings, config, attempted_performance=score)
        best_score = max(best_score, score)
        write_json(output / "best.json", {"steps": step, "config": sampler.recommend(),
                                          "performance": best_score, "unit": unit})
        print(f"   {score:.3f} {unit} (best so far {best_score:.3f})", flush=True)

    if not sampler.measured:
        raise ValueError(f"no build succeeded; see {output}")
    # Fastest of all builds measured in the session, including the final model prediction
    print(f"\nbest build ({best_score:.3f} {unit}): {describe(application, sampler.recommend())}")
    print(f"details: {output / 'best.json'}")
    return output
