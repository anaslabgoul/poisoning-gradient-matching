"""Verify exact minibatch continuation and the explicitly requested run scope."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.run_bilevel_trials import BilevelTrials, SliceExpired, make_config


class RunnerTests(unittest.TestCase):
    def test_frozen_scope(self):
        protocol = json.loads(Path('experiments/protocols/bilevel-resnet18-three-trials.json').read_text())
        config = make_config(argparse.Namespace(output='artifacts/test', data='data'), protocol)
        self.assertEqual(config.model, 'ResNet18')
        self.assertIsNone(config.train_size)
        self.assertEqual(len(config.cases)*(1+len(protocol['lambdas'])), 15)
        self.assertEqual(config.epochs, 40)
        self.assertEqual(config.inner_steps, 3)
        self.assertEqual(set(protocol['lambdas']), {1., .1, .001, .0001})

    def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rng(self):
        torch.manual_seed(27)
        initial = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                       torch.nn.Tanh(), torch.nn.Linear(4, 2))
        x, y = torch.rand(8, 3), torch.arange(8)%2

        def fixture(root, interrupt=False):
            instance = object.__new__(BilevelTrials)
            instance.out, instance.device = Path(root), torch.device('cpu')
            instance.model_name, instance.images, instance.labels = 'toy', x.clone(), y.clone()
            instance.dm, instance.ds = 0., 1.
            instance.config = SimpleNamespace(epochs=2, batch_size=4, radius=16., inner_steps=1,
                gradient_chunk_size=2, inner_step_size=16., checkpoint_seconds=100000., eps=16.)
            instance.args = SimpleNamespace(dataset='CIFAR10', epochs=2, noaugment=False,
                data_aug='default', optimization='conservative')
            # Exercise common augmentation RNG restoration, not just model state.
            instance.kettle = SimpleNamespace(augment=lambda images, randgen: (images+.01*randgen[:, :3]).clamp(0,1))
            instance.evaluate = lambda model: dict(clean_test_accuracy=float((model(x).argmax(1)==y).float().mean()))
            calls = [0]
            def stop(margin=180):
                calls[0] += 1
                return interrupt and calls[0]>=3
            instance.out_of_time = stop
            return instance

        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'full', Path(directory)/'resumed'
            # CPU fixture makes the resume check available without a GPU.
            patches = [patch('experiments.run_bilevel_trials.get_model', side_effect=lambda *args: copy.deepcopy(initial)),
                patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.max_memory_allocated', return_value=0),
                patch('torch.cuda.get_rng_state_all', return_value=[]), patch('torch.cuda.set_rng_state_all'),
                patch('torch.cuda.manual_seed_all')]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                fixture(a).train_condition('check', 201, .001)
                with self.assertRaises(SliceExpired):
                    fixture(b, interrupt=True).train_condition('check', 201, .001)
                saved = torch.load(b/'checkpoints/check.pt', weights_only=False)
                self.assertEqual(saved['offset'], 4)
                self.assertEqual(saved['epoch'], 0)
                fixture(b).train_condition('check', 201, .001)
                first = torch.load(a/'models/check.pt', weights_only=True)
                second = torch.load(b/'models/check.pt', weights_only=True)
                for key in first:
                    torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                one = json.loads((a/'runs/check.json').read_text())
                two = json.loads((b/'runs/check.json').read_text())
                self.assertEqual(one['history'], two['history'])


if __name__ == '__main__':
    unittest.main()
