"""Evaluation must observe sessions without changing their sampling decisions."""
import io
import tempfile
import json
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.active_sampling import LiveActiveSampler, _session, simulate
from ml.evaluation_trace import EvaluationTrace, validate_budget
from ml.evaluate import evaluate_sampling
from ml.load import load_table
from ml.models.residual_correction import ResidualCorrection
from ml.live_llama import GGUF_NAME, benchmark as run_benchmark, run_session, sampling_space


class ToyModel:
    def fit(self, table):
        self.scale = table.performance.max()
        return self

    def predict_performance(self, hardware, builds):
        return np.array([float(b['x']) + 1 for b in builds]) / self.scale, np.zeros(len(builds))


def pool():
    return pd.DataFrame({'application': ['llama.cpp'] * 6, 'platform': ['target'] * 6,
                         'cfg.x': list(map(str, range(6))), 'performance': np.arange(1., 7.),
                         'relative': np.arange(1., 7.) / 6})


class EvaluationTests(unittest.TestCase):
    def test_quality_separates_attempt_from_recommendation(self):
        trace = EvaluationTrace(2)
        trace.record(2, 2, {'x': 'B'}, .8, .8, predicted_performance=80.)
        trace.timing(2, 2, .1, .2, .3, 'failed', benchmark_seconds=120.,
                     attempted_build={'x': 'A'}, configure_seconds=1., compile_seconds=118.)
        row = trace.tables()['quality'][0]
        self.assertEqual(json.loads(row['attempted_build']), {'x': 'A'})
        self.assertEqual(json.loads(row['recommended_build']), {'x': 'B'})
        self.assertEqual(row['attempt_status'], 'failed')
        self.assertEqual(row['attempt_total_seconds'], 120.)
        self.assertEqual(row['recommended_predicted_performance'], 80.)
        self.assertIsNone(row['recommended_measured_performance'])
        self.assertIsNone(row['attempt_benchmark_execution_seconds'])
        self.assertFalse({'build', 'status', 'predicted_performance', 'actual_performance'} & row.keys())

    def test_benchmark_phase_timings_include_failure(self):
        for failed in (None, 'configure', 'compile', 'bench'):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'step'
                output.mkdir()
                timings = {}

                def execute(command, **kwargs):
                    if command[0] == failed:
                        raise subprocess.CalledProcessError(1, command)
                    if command[0] == 'bench':
                        json.dump([{'n_prompt': 512, 'n_gen': 0, 'avg_ns': 1e9,
                                    'backends': 'cpu', 'n_gpu_layers': 0},
                                   {'n_prompt': 0, 'n_gen': 128, 'avg_ns': 1e9,
                                    'backends': 'cpu', 'n_gpu_layers': 0}], kwargs['stdout'])

                with patch('ml.live_llama.commands', return_value=(['configure'], ['compile'], ['bench'], [])), \
                     patch('ml.live_llama.compiler_flags', return_value={}), \
                     patch('ml.live_llama.subprocess.run', side_effect=execute), \
                     patch('ml.live_llama.time.perf_counter', side_effect=range(6)):
                    if failed:
                        with self.assertRaises(RuntimeError):
                            run_benchmark({}, Path(tmp), Path(tmp), output, {}, 'cpu', timings=timings)
                    else:
                        self.assertEqual(run_benchmark({}, Path(tmp), Path(tmp), output, {}, 'cpu', timings=timings), 320.)
                last_phase = {'configure': 0, 'compile': 1, 'bench': 2, None: 2}[failed]
                record = json.loads((output / 'build.json').read_text())
                for index, key in enumerate(('configure_seconds', 'compile_seconds', 'benchmark_execution_seconds')):
                    self.assertEqual(timings[key], 1 if index <= last_phase else None)
                    self.assertEqual(record[key], timings[key])

    def test_historical_prediction_units(self):
        source = ToyModel().fit(pool())
        correction = ResidualCorrection(source, ToyModel, {}).fit(pool())
        self.assertEqual(correction.performance_scale, 6)
        means, _ = correction.predict_performance({}, [{'x': '5'}])
        self.assertAlmostEqual(means[0] * correction.performance_scale, 6)

    def test_invalid_json_is_only_skipped_when_requested(self):
        error = json.JSONDecodeError('bad benchmark', '', 0)
        good = dict(pool().iloc[0], run_id='good', workload='[[0, 128], [512, 0]]')
        with patch('ml.load.read_runs', return_value=[{}, {}]), patch('ml.load.flag_catalog'), patch('ml.load.parse_run', side_effect=[error, good]):
            self.assertEqual(len(load_table(['unused'], 'llama.cpp', skip_invalid_json=True)), 1)
        with patch('ml.load.read_runs', return_value=[{}]), patch('ml.load.flag_catalog'), patch('ml.load.parse_run', side_effect=error):
            with self.assertRaises(json.JSONDecodeError):
                load_table(['unused'], 'llama.cpp')

    def test_trace_detects_leaving_and_returning_to_optimum(self):
        trace = EvaluationTrace(2)
        for step, flag, actual in [(2, 'a', 10), (3, 'b', 8), (4, 'a', 10), (5, 'a', 10)]:
            trace.record(step, step, {'x': flag}, 1., .8, actual=actual, optimum=10.)
        tables = trace.tables()
        final = tables['final'][0]
        self.assertEqual(final['first_pool_best_round'], 1)
        self.assertEqual(final['stable_pool_best_round'], 3)
        self.assertFalse(final['sticks_to_pool_best'])
        self.assertEqual(tables['improvement'][1]['flags_changed_from_final'], 1)
        self.assertAlmostEqual(tables['improvement'][-1]['prediction_gain_fraction'], .25)

    def test_replay_predictions_do_not_change_choices(self):
        with patch.dict('ml.active_sampling.MODEL_TYPES', {'toy': ToyModel}):
            plain = _session('toy', pool(), 5, 2, 0)
            evaluated = _session('toy', pool(), 5, 2, 0, evaluation=True)
            startup_only = _session('toy', pool(), 2, 2, 0, evaluation=True)['_evaluation']
        self.assertEqual(startup_only['timings'], [])
        self.assertEqual(startup_only['quality'], [])
        self.assertEqual(startup_only['final'][0]['benchmark_attempts'], 2)
        for key in plain.keys() - {'seconds'}:
            if isinstance(plain[key], float) and np.isnan(plain[key]):
                self.assertTrue(np.isnan(evaluated[key]))
            else:
                self.assertEqual(plain[key], evaluated[key])
        tables = evaluated['_evaluation']
        self.assertEqual(len(tables['convergence']), 3)
        self.assertTrue(tables['final'][0]['final_is_pool_best'])
        self.assertEqual([row['step'] for row in tables['timings']], [3, 4, 5])
        self.assertEqual([row['round'] for row in tables['timings']], [1, 2, 3])
        self.assertEqual(tables['final'][0]['benchmark_attempts'], 5)
        for row in tables['timings']:
            self.assertEqual(row['attempted_measured_performance'], float(json.loads(row['attempted_build'])['x']) + 1)
            self.assertAlmostEqual(row['best_observed_percent_of_pool_best'], 100 * plain[f"best@{row['step']}"])
        self.assertTrue(all(row['prediction_seconds'] >= 0 for row in tables['timings']))
        quality = tables['quality']
        self.assertEqual([row['step'] for row in quality], [3, 4, 5])
        self.assertTrue(all(row['recommended_percent_of_pool_best'] == 100 for row in quality))
        self.assertTrue(all(row['recommended_predicted_performance'] == 6 for row in quality))
        self.assertTrue(all(row['recommended_prediction_absolute_error'] == 0 for row in quality))
        self.assertTrue(all(row['prediction_seconds'] >= 0 for row in quality))
        self.assertTrue(all(row['initialization_seconds'] > 0 for row in quality))
        self.assertTrue(all(row['attempt_benchmark_execution_seconds'] is None for row in quality))

    def test_budget_and_csv_grid(self):
        for steps in (1, 26):
            with self.assertRaises(ValueError):
                validate_budget(steps, 2)
        data = pd.concat([pool(), pool().assign(platform='history')])
        with tempfile.TemporaryDirectory() as tmp, patch.dict('ml.active_sampling.MODEL_TYPES', {'toy': ToyModel}), patch.dict('ml.evaluate.MODEL_TYPES', {'toy': ToyModel}):
            evaluate_sampling(data, tmp, models=['toy', 'random'], historical=['none', 'toy'],
                              steps=4, initial=2, seeds=3, seed=7, jobs=1)
            final = pd.read_csv(Path(tmp) / 'evaluation.final.csv')
            self.assertEqual(len(final), 14)  # two targets: 2 cold models x 3 seeds + 1 warm
            self.assertEqual(set(final[final.pretrained == 'toy'].seed), {7})
            self.assertEqual(set(final[final.pretrained == 'none'].seed), {7, 8, 9})
            self.assertEqual(len(list(Path(tmp).glob('*.csv'))), 5)
            random = final[final.model == 'random']
            self.assertTrue(random.final_prediction_value.isna().all())

    def test_simulate_repeats_only_cold_start(self):
        with patch.dict('ml.active_sampling.MODEL_TYPES', {'toy': ToyModel}):
            result = simulate(pool(), ['toy'], steps=4, seeds=3, jobs=1,
                              pretrained=ToyModel().fit(pool()))
        self.assertEqual(result.groupby('model').size().to_dict(),
                         {'corrected-toy': 1, 'random': 3, 'toy': 3})

    def test_live_failures_and_rng_isolation(self):
        # Small real sampler/search with synthetic benchmarks; no compiler or hardware needed.
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'llama.cpp').mkdir()
            (work / 'llama.cpp' / 'CMakeLists.txt').touch()
            (work / 'gguf').mkdir()
            (work / 'gguf' / GGUF_NAME).touch()
            choices = []

            def benchmark(config, *args, **kwargs):
                choices.append(config['x'])
                if len(choices) == 3:
                    raise RuntimeError('synthetic failure')
                return float(config['x']) + 1

            with patch('ml.live_llama.platform_name', return_value='target'), patch('ml.live_llama.hardware_features', return_value={}), patch('ml.live_llama.describe', return_value='toy'), patch('ml.live_llama.benchmark', side_effect=benchmark), patch('ml.search.valid', return_value=True), patch('ml.active_sampling.APPS') as apps:
                apps.__getitem__.return_value.HARDWARE_DETERMINED = ()
                output = run_session(ToyModel, {}, pool(), pretrain=lambda _: None, steps=5,
                                     initial=2, seed=0, device='cpu', workdir=tmp,
                                     values={'x': list(map(str, range(8)))}, evaluation=True)
                control = LiveActiveSampler(ToyModel, sampling_space({'x': list(map(str, range(8)))}, 'cpu'),
                                            {}, 'llama.cpp', initial=2, seed=0)
                expected = []
                for step in range(1, 6):
                    config = control.ask()
                    expected.append(config['x'])
                    if step == 3:
                        control.fail(config)
                    else:
                        control.tell(config, float(config['x']) + 1)
                self.assertEqual(choices, expected)
            timings = pd.read_csv(output / 'evaluation.timings.csv')
            final = pd.read_csv(output / 'evaluation.final.csv').iloc[0]
            self.assertEqual(len(choices), 5)
            self.assertEqual(len(set(choices)), 5)
            self.assertEqual(timings.attempt_status.tolist().count('failed'), 1)
            self.assertEqual(timings.step.tolist(), [3, 4, 5])
            self.assertTrue(pd.isna(timings.loc[0, 'attempted_measured_performance']))
            self.assertEqual(timings.loc[0, 'best_observed_performance'], max(float(choice) + 1 for choice in choices[:2]))
            self.assertEqual(timings.loc[1, 'attempted_measured_performance'], float(choices[3]) + 1)
            self.assertTrue(timings.pool_best_performance.isna().all())
            self.assertEqual(final.prediction_count, 3)
            self.assertEqual(final.final_step, 5)
            self.assertTrue(pd.isna(final.final_is_pool_best))
            self.assertTrue(np.isfinite(final.final_prediction_value))
            quality = pd.read_csv(output / 'evaluation.quality.csv')
            self.assertEqual(quality.step.tolist(), [3, 4, 5])
            np.testing.assert_allclose(quality.recommended_predicted_performance, 8)
            self.assertTrue((quality.performance_unit == 'tokens/s').all())
            self.assertTrue(quality.recommended_measured_performance.isna().all())  # predicted winner was never measured
            self.assertTrue(quality.recommended_percent_of_pool_best.isna().all())
            self.assertTrue((quality.prediction_seconds > 0).all())
            self.assertTrue((quality.attempt_total_seconds >= 0).all())
            self.assertTrue(quality.session_elapsed_seconds.is_monotonic_increasing)
            self.assertTrue((quality.session_elapsed_seconds >= quality.initialization_seconds).all())
            for row in quality.itertuples():
                self.assertEqual(json.loads(row.attempted_build)['x'], choices[row.step - 1])
                self.assertEqual(json.loads(row.recommended_build)['x'], '7')
            self.assertEqual([json.loads(build)['x'] for build in timings.attempted_build], choices[2:])

    def test_live_builds_final_prediction_last(self):
        # The last build of the budget is the model's final prediction if it was not built yet
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'llama.cpp').mkdir()
            (work / 'llama.cpp' / 'CMakeLists.txt').touch()
            (work / 'gguf').mkdir()
            (work / 'gguf' / GGUF_NAME).touch()
            choices = []

            def benchmark(config, *args, **kwargs):
                choices.append(config['x'])
                return float(config['x']) + 1

            original = LiveActiveSampler.predict

            def predict(self):
                # Unbuilt prediction: the toy model's pick with x replaced by 1 (seed 0 builds 6, 0, 7 first)
                config, score = original(self)
                return {**config, 'x': '1'}, score

            stdout = io.StringIO()
            with patch('ml.live_llama.platform_name', return_value='target'), patch('ml.live_llama.hardware_features', return_value={}), patch('ml.live_llama.describe', side_effect=lambda app, config: config['x']), patch('ml.live_llama.benchmark', side_effect=benchmark), patch('ml.search.valid', return_value=True), patch('ml.active_sampling.APPS') as apps, \
                 patch.object(LiveActiveSampler, 'predict', predict), patch('sys.stdout', stdout):
                apps.__getitem__.return_value.HARDWARE_DETERMINED = ()
                output = run_session(ToyModel, {}, pool(), pretrain=lambda _: None, steps=4,
                                     initial=2, seed=0, device='cpu', workdir=tmp,
                                     values={'x': list(map(str, range(8)))}, evaluation=True)
            self.assertEqual(choices, ['6', '0', '7', '1'])
            self.assertIn('-- build 4/4 (final model prediction): 1', stdout.getvalue())
            # Only one result: the fastest measured build, not an unmeasured model pick
            self.assertIn('best build (8.000 tokens/s): 7', stdout.getvalue())
            self.assertNotIn('model pick', stdout.getvalue())
            self.assertEqual(json.loads((output / 'best.json').read_text())['config']['x'], '7')


if __name__ == '__main__':
    unittest.main()
