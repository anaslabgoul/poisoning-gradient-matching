"""Prevent cohort expansion and misleading comparisons of unfinished subsets."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.compare_saved_gradient import aggregate,frozen_cohort,resolve_lambda,run


class SavedGradientComparisonTest(unittest.TestCase):
    def test_lambda_override_and_calibrated_default(self):
        self.assertEqual(resolve_lambda(.001393,None),.001393)
        self.assertEqual(resolve_lambda(.001393,.3),.3)
        self.assertEqual(resolve_lambda(.001393,0.),0.)
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                resolve_lambda(.001393,value)

    def rows(self):
        return [dict(run_index=i,target_id=100+i//2,victim_seed=11001+i%2,
            poison_file_sha256='hash-'+str(i),attack_success=s,
            clean_test_accuracy=a,true_label=0,intended_label=1,target={'target_id':100+i//2})
            for i,(s,a) in enumerate([(1,.8),(0,.82),(1,.79),(0,.81)])]

    def test_partial_uses_only_matched_baseline_and_full_tracks_regressions(self):
        base=self.rows()
        cohort=frozen_cohort({'records':base},expected=4)
        defense=[dict(r,attack_success=s,clean_test_accuracy=a) for r,s,a in
            zip(base,[0,1,0,0],[.78,.81,.80,.81])]
        partial=aggregate(cohort,defense[:1])
        self.assertEqual(partial['paired']['baseline_asr'],1.)
        self.assertEqual(partial['paired']['asr_reduction_pp'],100.)
        self.assertFalse(partial['complete'])
        full=aggregate(cohort,defense)
        self.assertTrue(full['complete'])
        self.assertEqual(full['paired']['baseline_successes'],2)
        self.assertEqual(full['paired']['defended_successes'],1)
        self.assertEqual(full['paired']['attacks_prevented'],2)
        self.assertEqual(full['paired']['attacks_introduced'],1)
        self.assertEqual(full['paired']['asr_reduction_pp'],25.)
        self.assertAlmostEqual(full['paired']['accuracy_change_pp'],-.5)

    def test_rejects_extra_or_mismatched_results(self):
        rows=self.rows()
        cohort=frozen_cohort({'records':rows},expected=4)
        with self.assertRaises(ValueError):
            frozen_cohort({'records':rows+rows[:1]},expected=4)
        with self.assertRaises(ValueError):
            aggregate(cohort,[dict(rows[0],victim_seed=7)])
        with self.assertRaises(ValueError):
            aggregate(cohort,[rows[0],rows[0]])

    def test_launch_refuses_a_hundred_run_cohort(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'cohort.json'
            path.write_text(json.dumps({'expected':100}))
            with patch('experiments.compare_saved_gradient.Comparison') as comparison:
                with self.assertRaisesRegex(ValueError,'limited to the 25'):
                    run(SimpleNamespace(cohort=str(path)))
                comparison.assert_not_called()


if __name__=='__main__':
    unittest.main()
