"""Frozen cache identity and the lambda-only experiment scope."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch

from experiments.run_small_lambda import SmallLambdaTrials,validate_protocol,verify_inputs


class SmallLambdaTests(unittest.TestCase):
    def test_requested_scope_keeps_training_and_optimizer_unchanged(self):
        p=json.loads(Path('experiments/protocols/bilevel-small-lambda.json').read_text())
        validate_protocol(p)
        self.assertEqual(p['lambdas'],[1e-8,1e-10])
        self.assertEqual(p['new_victim_trainings'],2)
        self.assertEqual(p['reused_undefended_models'],1)
        self.assertEqual(p['epochs'],40)
        self.assertEqual(p['train_images'],50000)
        self.assertEqual(p['model'],'ResNet18')
        self.assertEqual(p['optimizer']['learning_rate'],.1)
        self.assertEqual(p['optimizer']['momentum'],.9)
        for name in ('clipping','warmup','adaptive_lambda','update_rescaling'):
            self.assertFalse(p['optimizer'][name])

    def test_new_batch_values_and_counts_are_validated(self):
        p=json.loads(Path('experiments/protocols/bilevel-lambda-7-6-5.json').read_text())
        validate_protocol(p)
        self.assertEqual(p['lambdas'],[1e-7,1e-6,1e-5])
        self.assertEqual(p['new_victim_trainings'],3)
        self.assertEqual(p['victim_models'],4)
        p['new_victim_trainings']=2
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_cached_runner_attempts_only_the_three_requested_models(self):
        p=json.loads(Path('experiments/protocols/bilevel-lambda-7-6-5.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            trial=object.__new__(SmallLambdaTrials)
            trial.out=Path(directory)
            trial.source=Path(directory)/'cached'
            trial.protocol=p
            trial.baseline=dict(name='trial-0-standard')
            trial.cases=[dict(key=3800000000,target_id=7171,poison_ids=[4,7])]
            trial.config=SimpleNamespace(seeds=[18001])
            trial.summarize_trials=Mock(return_value=dict(all_succeeded=True,attempted=4))
            trial.train_condition=Mock()
            with patch('experiments.run_small_lambda.load_state',return_value=dict(
                    poison_ids=torch.tensor([4,7]),target_id=7171,delta=torch.zeros(2))):
                trial.run_cached()
            self.assertEqual([c.args[2] for c in trial.train_condition.call_args_list],[1e-7,1e-6,1e-5])
            status=json.loads((trial.out/'status.json').read_text())
            self.assertEqual(status['expected'],4)
            self.assertEqual(status['new_defended_trainings'],3)
            self.assertEqual(status['cached_baselines'],1)

    def test_cached_assets_are_verified_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in ('runs','models','poisons'):
                (root/path).mkdir()
            (root/'models/trial-0-standard.pt').write_bytes(b'original model')
            (root/'poisons/3800000000.pt').write_bytes(b'original poison')
            poison_hash=hashlib.sha256(b'original poison').hexdigest()
            config=dict(dataset='CIFAR10',model='ResNet18',train_size=None,epochs=40,
                batch_size=128,cases=[3800000000],seeds=[18001],precision='fp32',
                eps=16.,budget=.01,radius=16.,inner_steps=3,regularizer_samples=8)
            (root/'manifest.json').write_text(json.dumps(dict(config=config)))
            baseline=dict(status='complete',seed=18001,case_key=3800000000,poison_sha256=poison_hash)
            (root/'runs/trial-0-standard.json').write_text(json.dumps(baseline))
            files=('manifest.json','runs/trial-0-standard.json','models/trial-0-standard.pt','poisons/3800000000.pt')
            protocol=dict(case_keys=[3800000000],victim_seeds=[18001],
                frozen_files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
            _,record=verify_inputs(protocol,root)
            self.assertEqual(record,baseline)
            (root/'models/trial-0-standard.pt').write_bytes(b'changed model')
            with self.assertRaisesRegex(RuntimeError,'Frozen source file changed'):
                verify_inputs(protocol,root)


if __name__=='__main__':
    unittest.main()
