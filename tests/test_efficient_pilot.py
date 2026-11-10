"""Keep natural mistakes separate and prevent unintended case/seed products."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

from experiments.compare_defenses import METHODS
from experiments.search_efficient_conditions import configuration,summarize
from experiments.run_efficient_defense_study import run


class EfficientPilotTest(unittest.TestCase):
    def test_summary_preserves_natural_errors_and_attribution(self):
        rows=[]
        for success,natural,correct in [(1,0,1),(0,0,1),(1,1,0)]:
            rows.append(dict(poison_target={'attack_success':success},
                clean_target={'attack_success':natural,'target_correct':correct},
                clean_accuracy=.8,poison_accuracy=.79,victim_training_seconds=60,
                poison_generation_seconds=30,peak_gpu_bytes=1024**3,
                poison_generation_peak_gpu_bytes=2*1024**3))
        result=summarize(rows)
        self.assertEqual(result['trials'],3)
        self.assertEqual(result['successes'],2)
        self.assertEqual(result['clean_control_intended_successes'],1)
        self.assertEqual(result['attributable_successes'],1)
        self.assertEqual(result['peak_gpu_gib'],2.)

    def test_unqualified_pilot_cannot_start_training(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'conditions.json'
            path.write_text(json.dumps({'qualified':False}))
            with patch('experiments.run_efficient_defense_study.Comparison') as comparison:
                with self.assertRaisesRegex(RuntimeError,'not passed'):
                    run(argparse.Namespace(conditions=str(path)),Path(directory))
                comparison.assert_not_called()

    def test_two_cases_mean_eight_victims_not_a_seed_cross_product(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            settings=argparse.Namespace(data='.',model='ResNet20',train_size=10000,
                epochs=60,budget=.04,loss='similarity',restarts=4)
            selected=vars(configuration(settings,root/'main',[4000000000,4000001009]))
            selected['qualified']=True
            path=root/'conditions.json'
            path.write_text(json.dumps(selected))
            fake=Mock()
            fake.cases=[dict(key=4000000000),dict(key=4000001009)]
            fake.manifest={'git_commit':'test'}
            fake.fit.return_value=({'full_test_accuracy':.8},{})
            fake.calibrate.return_value={m:0. for m in METHODS}
            with patch('experiments.run_efficient_defense_study.Comparison',return_value=fake), \
                 patch('experiments.run_efficient_defense_study.statistics'):
                run(argparse.Namespace(conditions=str(path),trials=2,data='.'),root/'study')
            self.assertEqual(fake.fit.call_count,9)
            victims=fake.fit.call_args_list[1:]
            for index,call in enumerate(victims):
                self.assertEqual(call.args[2],70000+index//4)
                self.assertEqual(call.args[1],METHODS[index%4])
            status=json.loads((root/'study/status.json').read_text())
            self.assertEqual(status['victim_trainings'],8)


if __name__=='__main__':
    unittest.main()
