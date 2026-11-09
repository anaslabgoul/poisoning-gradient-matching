"""The four-model workflow must not require extra clean-control trainings."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.compare_defenses import Comparison, METHODS


class FourModelSummaryTest(unittest.TestCase):
    def test_four_completed_victims_are_complete_without_clean_controls(self):
        with TemporaryDirectory() as directory:
            experiment = Comparison.__new__(Comparison)
            experiment.out = Path(directory)
            experiment.config = SimpleNamespace(cases=[7], seeds=[11], clean_controls=False)
            experiment.manifest = {'evaluation': 'fixed poisons'}
            (experiment.out/'runs').mkdir()
            for method in METHODS:
                record = dict(name=method, method=method, poisoned=True, case_key=7,
                              attack_success=int(method == 'standard'),
                              clean_test_accuracy=.9, train_seconds=1., peak_gpu_bytes=1024)
                (experiment.out/'runs'/(method+'.json')).write_text(json.dumps(record))
            result = experiment.summarize()
            self.assertTrue(result['complete'])
            self.assertEqual(result['expected_clean_runs_per_method'], 0)
            self.assertEqual(len(result['summary']), 4)
            self.assertTrue(all(row['clean_control_accuracy'] is None for row in result['summary']))


if __name__ == '__main__':
    unittest.main()
