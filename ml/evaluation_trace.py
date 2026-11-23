"""Per-session evaluation of sampling and recommendation measurements.

Prediction scores are model-native (not necessarily calibrated across refits).
Improvement therefore compares the current pick and the first pick under the
SAME current model. Simulation additionally reports measured pool performance.
"""
import csv
import json
from pathlib import Path

import numpy as np


MAX_BUDGET = 25
PREDICTION_TIMING_FIELDS = (
    'prediction_measurement_status', 'prediction_measurement_seconds',
    'prediction_configure_seconds', 'prediction_compile_seconds',
    'prediction_benchmark_execution_seconds',
)


def validate_budget(steps, initial):
    if not 2 <= initial <= steps <= MAX_BUDGET:
        raise ValueError('require 2 <= initial <= steps <= 25 (benchmark attempts)')


def build_json(config):
    return json.dumps(config, sort_keys=True, default=str)


def distance(left, right):
    return sum(left.get(key) != right.get(key) for key in left.keys() | right.keys())


class EvaluationTrace:
    def __init__(self, initial, *, include_startup=False):
        self.initial = initial
        self.include_startup = include_startup
        self.picks = []
        self.timings = []
        self.initialization_seconds = 0.0

    def timing(self, step, successful, ask, refit, prediction, status='completed',
               benchmark_seconds=None, elapsed_seconds=None, configure_seconds=None,
               compile_seconds=None, benchmark_execution_seconds=None, attempted_build=None,
               attempted_performance=None, best_observed=None, optimum=None, unit=None,
               best_build=None, prediction_measurement=None):
        successful_before = self.timings[-1]['successful_builds'] if self.timings else 0
        measurement = prediction_measurement or {}
        self.timings.append(dict(step=step, round=self.picks[-1]['round'] if self.picks else 0,
                                 phase='startup' if successful_before < self.initial else 'evaluation',
                                 attempted_build=None if attempted_build is None else build_json(attempted_build),
                                 attempted_measured_performance=attempted_performance,
                                 best_observed_performance=best_observed, pool_best_performance=optimum,
                                 best_observed_percent_of_pool_best=None if optimum is None or best_observed is None
                                 else 100 * best_observed / optimum, performance_unit=unit,
                                 attempt_status=status, successful_builds=successful, next_sample_seconds=ask,
                                 refit_seconds=refit, prediction_seconds=prediction,
                                 initialization_seconds=self.initialization_seconds,
                                 attempt_configure_seconds=configure_seconds, attempt_compile_seconds=compile_seconds,
                                 attempt_benchmark_execution_seconds=benchmark_execution_seconds,
                                 attempt_total_seconds=benchmark_seconds, session_elapsed_seconds=elapsed_seconds,
                                 best_observed_build=None if best_build is None else build_json(best_build),
                                 **{field: measurement.get(field) for field in PREDICTION_TIMING_FIELDS}))

    def record(self, step, successful, config, score, first_score, *, actual=None, optimum=None,
               best_before, predicted_performance=None, unit=None):
        before_build, before_performance = best_before
        improved = actual is not None and actual > before_performance
        output_build, output_performance = (config, actual) if improved else best_before
        output_source = 'prediction' if improved else 'active_sampling'
        self.picks.append(dict(step=step, round=len(self.picks) + int(not self.include_startup),
                               successful_builds=successful, config=dict(config), predicted_value=score,
                               first_build_current_prediction=first_score,
                               actual_performance=actual, optimum=optimum, best_observed=before_performance,
                               best_before_prediction_build=build_json(before_build),
                               output_build=build_json(output_build),
                               output_measured_performance=output_performance,
                               output_source=output_source,
                               predicted_performance=predicted_performance, performance_unit=unit))
        return output_build, output_performance, output_source

    def tables(self):
        convergence, improvement, quality = [], [], []
        timings = {row['step']: row for row in self.timings}
        first = self.picks[0] if self.picks else None
        final = self.picks[-1] if self.picks else None
        optimal = []
        for i, pick in enumerate(self.picks):
            config = pick['config']
            actual, optimum = pick['actual_performance'], pick['optimum']
            is_best = None if actual is None or optimum is None else bool(np.isclose(actual, optimum, rtol=1e-9, atol=0))
            optimal.append(is_best)
            base = dict(step=pick['step'], round=pick['round'], successful_builds=pick['successful_builds'],
                        recommended_build=build_json(config))
            convergence.append(dict(base, changed_from_previous=False if i == 0 else config != self.picks[i - 1]['config'],
                                    is_pool_best=is_best, same_as_final=config == final['config']))
            first_score, score = pick['first_build_current_prediction'], pick['predicted_value']
            timing = timings.get(pick['step'], {})
            quality.append(dict(base, recommended_predicted_value=score,
                                recommended_predicted_performance=pick['predicted_performance'],
                                performance_unit=pick['performance_unit'],
                                recommended_measured_performance=actual,
                                best_observed_performance=pick['best_observed'],
                                pool_best_performance=optimum,
                                recommended_percent_of_pool_best=None if optimum is None else 100 * actual / optimum,
                                recommended_percent_of_best_observed=None if actual is None or pick['best_observed'] is None
                                else 100 * actual / pick['best_observed'],
                                recommended_prediction_absolute_error=None if actual is None or pick['predicted_performance'] is None
                                else abs(pick['predicted_performance'] - actual),
                                recommended_is_pool_best=is_best,
                                best_before_prediction_build=pick['best_before_prediction_build'],
                                best_before_prediction_performance=pick['best_observed'],
                                output_build=pick['output_build'],
                                output_measured_performance=pick['output_measured_performance'],
                                output_source=pick['output_source'],
                                **{key: timing.get(key) for key in (
                                    'attempted_build', 'attempt_status', 'prediction_seconds', 'next_sample_seconds', 'refit_seconds',
                                    'initialization_seconds', 'attempt_configure_seconds', 'attempt_compile_seconds',
                                    'attempt_benchmark_execution_seconds',
                                    'attempt_total_seconds', 'session_elapsed_seconds', *PREDICTION_TIMING_FIELDS)}))
            improvement.append(dict(base, flags_changed_from_first=distance(config, first['config']),
                                    flags_changed_from_final=distance(config, final['config']),
                                    predicted_value=score, first_build_current_prediction=first_score,
                                    prediction_gain=None if score is None else score - first_score,
                                    prediction_gain_fraction=None if not first_score else score / first_score - 1,
                                    actual_performance=actual,
                                    actual_gain_from_first=None if actual is None or first['actual_performance'] is None
                                    else actual - first['actual_performance'],
                                    actual_fraction_of_best=None if optimum is None else actual / optimum))
        first_best = next((p['step'] for p, best in zip(self.picks, optimal) if best), None)
        stable_best = next((p['step'] for i, p in enumerate(self.picks) if optimal[i] and all(optimal[i:])), None)
        stable_final = next((p['step'] for i, p in enumerate(self.picks)
                             if all(q['config'] == final['config'] for q in self.picks[i:])), None)
        last = self.timings[-1] if self.timings else {}
        final_output = final if final and final['step'] == last.get('step') else {}
        sampled_build = last.get('best_observed_build')
        summary = dict(benchmark_attempts=len(self.timings), prediction_count=len(self.picks), final_step=last.get('step'),
                       final_build=build_json(final['config']) if final else None,
                       final_prediction_value=final['predicted_value'] if final else None,
                       final_actual_performance=final['actual_performance'] if final else None,
                       pool_best_performance=last.get('pool_best_performance'),
                       final_is_pool_best=optimal[-1] if optimal else None,
                       first_pool_best_step=first_best, stable_pool_best_step=stable_best,
                       sticks_to_pool_best=None if first_best is None else first_best == stable_best,
                       stable_final_build_step=stable_final,
                       first_pool_best_round=None if first_best is None else next(p['round'] for p in self.picks if p['step'] == first_best),
                       stable_pool_best_round=None if stable_best is None else next(p['round'] for p in self.picks if p['step'] == stable_best),
                       stable_final_build_round=None if stable_final is None else next(p['round'] for p in self.picks if p['step'] == stable_final),
                       prediction_benchmark_attempts=sum(
                           row['prediction_measurement_status'] in ('completed', 'failed', 'replayed') for row in self.timings),
                       final_output_build=final_output.get('output_build', sampled_build),
                       final_output_measured_performance=final_output.get(
                           'output_measured_performance', last.get('best_observed_performance')),
                       final_output_source=final_output.get(
                           'output_source', 'active_sampling' if sampled_build is not None else None))
        reported_timings = [row for row in self.timings
                            if self.include_startup or row['phase'] == 'evaluation'
                            or row['prediction_measurement_status'] is not None]
        return dict(convergence=convergence, improvement=improvement, quality=quality,
                    final=[summary], sampling=self.timings, timings=reported_timings)


def save_tables(prefix, tables, identity, *, append=False):
    """Write the session summary, sampling history and per-round evaluation CSVs."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not rows:
            continue
        path = prefix.with_name(prefix.name + '.' + name + '.csv')
        rows = [{**identity, **row} for row in rows]
        header = not append or not path.exists() or path.stat().st_size == 0
        with path.open('a' if append else 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            if header:
                writer.writeheader()
            writer.writerows(rows)
