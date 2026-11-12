"""Frozen cache identity and the lambda-only experiment scope."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_small_lambda import verify_inputs


class SmallLambdaTests(unittest.TestCase):
    def test_requested_scope_keeps_training_and_optimizer_unchanged(self):
        p=json.loads(Path('experiments/protocols/bilevel-small-lambda.json').read_text())
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
