"""Workloads must be selected before measurements are combined or normalized."""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ml.load import load_table
from ml.main import main


def run(run_id, prompt=512, generate=128, seconds=1., flag='A', platform='target', phases=None):
    benchmark = phases or [dict(n_prompt=prompt, n_gen=0, avg_ns=seconds * 1e9),
                           dict(n_prompt=0, n_gen=generate, avg_ns=seconds * 1e9)]
    return {'source.yml': 'project_name: llama.cpp\n',
            'system.json': json.dumps({'CPU Info': {'Model Name': platform}}),
            'build.json': json.dumps({'run_id': run_id, 'config': {'x': flag}}),
            'benchmark.json': json.dumps(benchmark)}


class WorkloadTests(unittest.TestCase):
    def load(self, runs, **kwargs):
        with patch('ml.load.read_runs', return_value=runs), patch('ml.load.flag_catalog'), \
             patch('ml.load.hardware_features', return_value={}), \
             patch('ml.load.load_llama.build_config', side_effect=lambda b: b['config']):
            return load_table(['unused'], 'llama.cpp', **kwargs)

    def test_filter_precedes_deduplication_averaging_and_normalization(self):
        runs = [run('same-id', 32, 32, seconds=.001), run('same-id'),
                run('repeat', seconds=2.), run('other', seconds=2., flag='B'),
                run('excluded-target', 128, 1024, platform='other')]
        output = io.StringIO()
        with redirect_stdout(output):
            table = self.load(runs).set_index('cfg.x')
        self.assertEqual(set(table.platform), {'target'})
        self.assertEqual(table.loc['A', 'performance'], 240.)  # average of 320 and 160
        self.assertEqual(table.loc['A', 'runs'], 2)
        self.assertEqual(table.loc['A', 'relative'], 1.)
        self.assertAlmostEqual(table.loc['B', 'relative'], 160 / 240)
        self.assertEqual(set(table.workload), {'[[0, 128], [512, 0]]'})
        self.assertIn('included 3, excluded 1', output.getvalue())
        self.assertIn('included 0, excluded 1', output.getvalue())

    def test_custom_workload(self):
        table = self.load([run('default'), run('short', 32, 32)], prompt=32, generate=32)
        self.assertEqual(table.run_id.tolist(), ['short'])
        self.assertEqual(table.workload.tolist(), ['[[0, 32], [32, 0]]'])

    def test_exact_phases_required(self):
        combined = [dict(n_prompt=512, n_gen=128, avg_ns=1e9)]
        duplicate = [dict(n_prompt=512, n_gen=0, avg_ns=1e9)] * 2 + [dict(n_prompt=0, n_gen=128, avg_ns=1e9)]
        for phases in (combined, duplicate):
            with self.subTest(phases=phases), self.assertRaisesRegex(ValueError, 'no llama.cpp runs match'):
                self.load([run('bad', phases=phases)])

    def test_empty_selection_and_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, 'no llama.cpp runs match workload pp512/tg128'):
            self.load([run('short', 32, 32)])
        with self.assertRaisesRegex(ValueError, 'positive token counts'):
            self.load([], prompt=0)

    def test_gromacs_is_not_filtered(self):
        row = {'application': 'gromacs', 'platform': 'target', 'run_id': 'g',
               'cfg.x': 'A', 'performance': 10.}
        with patch('ml.load.read_runs', return_value=[{}]), patch('ml.load.parse_run', return_value=row), \
             patch('ml.load.flag_catalog'):
            table = load_table(['unused'], 'gromacs', prompt=32, generate=32)
        self.assertEqual(table.performance.tolist(), [10.])

    def test_evaluate_passes_workload_to_loader(self):
        with patch('sys.argv', ['ml.main', 'evaluate', '--prompt', '32', '--generate', '32']), \
             patch('ml.main.load_table') as loader, patch('ml.main.evaluate_sampling'), \
             patch('ml.main.reset_warning_log'):
            main()
        self.assertEqual(loader.call_args.kwargs['prompt'], 32)
        self.assertEqual(loader.call_args.kwargs['generate'], 32)

    def test_live_rejects_nonmatching_workload_before_loading(self):
        for command in (['active-sampling', '--steps', '3'],
                        ['evaluate', '--mode', 'live', '--model', 'hardware', '--pretrained', 'none']):
            with self.subTest(command=command), patch('sys.argv', ['ml.main', *command, '--prompt', '32']), \
                 patch('ml.main.load_table') as loader, patch('sys.stderr', new_callable=io.StringIO), \
                 self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()

    def test_live_filters_history_before_starting_session(self):
        with patch('sys.argv', ['ml.main', 'active-sampling', '--steps', '3']), \
             patch('ml.main.discover', return_value={}), patch('ml.main.load_table') as loader, \
             patch('ml.main.run_session'), patch('ml.main.reset_warning_log'):
            main()
        self.assertEqual(loader.call_args.kwargs['prompt'], 512)
        self.assertEqual(loader.call_args.kwargs['generate'], 128)


if __name__ == '__main__':
    unittest.main()
