"""Check partial paired summaries and prevent mixing coefficient outcomes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.compare_defenses import write_json
from experiments.compare_saved_gradient import frozen_cohort
from experiments.fixed_target_grid import sha256
from experiments.summarize_lambda_sweep import collect


class LambdaSweepTest(unittest.TestCase):
    def test_partial_pairs_weight_validation_and_reference_integrity(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            base=[dict(run_index=i,target_id=10+i,victim_seed=20+i,
                poison_file_sha256=str(i),attack_success=s,clean_test_accuracy=.8,
                true_label=0,intended_label=1,target={'target_id':10+i}) for i,s in enumerate([1,0])]
            cohort=frozen_cohort({'records':base},expected=2)
            cohort_path=root/'cohort.json'
            write_json(cohort_path,cohort)
            reference=root/'reference.json'
            write_json(reference,{'records':[dict(r,lambda_gradient=.001) for r in base]})
            plan=dict(runs_per_lambda=2,expected_new_runs=4,
                settings=[dict(index=i,**{'lambda':v},directory='lambda-'+str(i)) for i,v in enumerate([0.,.1])],
                existing_reference={'lambda':.001,'comparison':str(reference),'comparison_sha256':sha256(reference)},
                interpretation='Exploratory')
            plan_path=root/'plan.json'
            write_json(plan_path,plan)
            first=root/'lambda-0/completed/0.json'
            write_json(first,dict(base[0],attack_success=0,lambda_gradient=0.))
            partial=collect(plan_path,root,cohort_path)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['completed_new_runs'],1)
            self.assertEqual(partial['settings'][0]['paired']['baseline_asr'],1.)
            self.assertEqual(partial['settings'][0]['paired']['asr_reduction_pp'],100.)
            self.assertEqual(partial['settings'][1]['status']['phase'],'pending')
            write_json(first,dict(base[0],lambda_gradient=.1))
            with self.assertRaisesRegex(ValueError,'different lambda'):
                collect(plan_path,root,cohort_path)
            write_json(first,dict(base[0],lambda_gradient=0.))
            for i in range(2):
                for row in base:
                    write_json(root/('lambda-'+str(i))/'completed'/(str(row['run_index'])+'.json'),
                               dict(row,lambda_gradient=plan['settings'][i]['lambda']))
            complete=collect(plan_path,root,cohort_path)
            self.assertTrue(complete['complete'])
            self.assertEqual(complete['completed_new_runs'],4)
            reference.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Original lambda result changed'):
                collect(plan_path,root,cohort_path)


if __name__=='__main__':
    unittest.main()
