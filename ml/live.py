"""Live active sampling on this machine (ml.main active-sampling), for any application.

run_session measures the sampler's next build and independently verifies the
model's recommendation. Sampling results can later load as training data.

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
from tempfile import TemporaryDirectory

from .active_sampling import LiveActiveSampler
from .evaluation_trace import EvaluationTrace, save_tables, validate_budget
from .load import CFG, EVALUATION_ONLY, app_options, columns, describe, hardware_features, platform_name, strip
from .models.common import unseen_count


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
    """Run `steps` sampling attempts, plus prediction checks for fitted models.

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
    identity = dict(application=application, platform=platform, pretrained=source,
                    unseen_values=None if pretrained is None else unseen_count(pretrained.values, values),
                    model=model, seed=seed,
                    workload=app.workload(context))

    session_start = time.perf_counter()
    sampler = LiveActiveSampler(model_type, values, hardware, application,
                                initial=initial, seed=seed, pretrained=pretrained)
    trace.initialization_seconds = time.perf_counter() - session_start

    def measure(config, build_dir, step_output):
        started = time.perf_counter()
        timings = {}
        try:
            score = app.benchmark(config, context, build_dir, step_output, system, device, timings=timings)
            status = 'completed'
        except RuntimeError as error:
            print(f"   failed: {error}", flush=True)
            score, status = None, 'failed'
        return score, status, time.perf_counter() - started, timings

    def measure_prediction(step, best_before):
        tick = time.perf_counter()
        # Fixed evaluation RNG, isolated from active candidate generation.
        state = sampler.search.rng.getstate()
        try:
            sampler.search.rng.seed(seed)
            config, score = sampler.predict()
            first = trace.picks[0]['config'] if trace.picks else config
            first_scores, _ = sampler.model.predict_performance(hardware, [first])
        finally:
            sampler.search.rng.setstate(state)
        predicted = score * getattr(sampler.model, 'performance_scale', best_before[1])
        prediction_seconds = time.perf_counter() - tick
        print(f"   prediction: {describe(application, config)} "
              f"({predicted:.3f} {unit} estimated in {prediction_seconds:.6f} s)", flush=True)
        prediction_output = output / f"{step:04d}-prediction"
        prediction_output.mkdir()
        (prediction_output / EVALUATION_ONLY).touch()
        # Use a clean build tree so verification cannot change later sampling or prediction builds.
        build.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"{session}-prediction-", dir=build.parent) as prediction_build:
            actual, status, elapsed, timings = measure(config, Path(prediction_build), prediction_output)
        best = trace.record(step, len(sampler.measured), config, score, float(first_scores[0]),
                            actual=actual, best_before=best_before, predicted_performance=predicted, unit=unit)
        measurement = dict(prediction_measurement_status=status, prediction_measurement_seconds=elapsed,
                           **{f"prediction_{name}": seconds for name, seconds in timings.items()})
        return best, prediction_seconds, measurement

    best_config, best_score = None, None
    for step in range(1, steps + 1):
        successful_before = len(sampler.measured)
        tick = time.perf_counter()
        config = sampler.ask()
        ask_seconds = time.perf_counter() - tick
        print(f"\n-- build {step}/{steps} ({sampler.source}): {describe(application, config)}", flush=True)
        step_output = output / f"{step:04d}"
        step_output.mkdir()
        score, status, benchmark_seconds, attempt_timings = measure(config, build, step_output)
        refit_seconds = 0.0
        if status == 'completed':
            tick = time.perf_counter()
            sampler.tell(config, score)
            if sampler.model is not None:
                refit_seconds = time.perf_counter() - tick
        else:
            sampler.fail(config)
        best_sample = max(sampler.measured.values(), key=lambda row: row[1], default=(None, None))
        best_config, best_score = best_sample
        output_source = 'active_sampling'
        prediction_seconds = 0.0
        measurement = {}
        if len(sampler.measured) >= initial and (successful_before >= initial or step == steps):
            if sampler.model is not None:
                prediction_output, prediction_seconds, measurement = measure_prediction(step, best_sample)
                if step == steps:
                    best_config, best_score, output_source = prediction_output
            else:
                trace.record(step, len(sampler.measured), best_config, None, None, actual=best_score,
                             best_before=best_sample, unit=unit)
        trace.timing(step, len(sampler.measured), ask_seconds, refit_seconds, prediction_seconds, status,
                     benchmark_seconds=benchmark_seconds, elapsed_seconds=time.perf_counter() - session_start,
                     attempted_build=config, attempted_performance=score, best_build=best_sample[0],
                     best_observed=best_sample[1], unit=unit, prediction_measurement=measurement, **attempt_timings)
        save_tables(output / 'evaluation', trace.tables(), identity)
        if best_config is not None:
            write_json(output / "best.json", {"steps": step, "config": best_config,
                                              "performance": best_score, "unit": unit,
                                              "output_source": output_source})
            print(f"   best so far: {best_score:.3f} {unit}", flush=True)

    if not sampler.measured:
        raise ValueError(f"no build succeeded; see {output}")
    # The final output compares the best sampled build with the last prediction.
    print(f"\nbest build ({best_score:.3f} {unit}): {describe(application, best_config)}")
    print(f"details: {output / 'best.json'}")
    return output
