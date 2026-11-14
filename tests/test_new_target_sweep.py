"""Confidence math, observer invariance, fixed scope and fresh-poison identity."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.compare_defenses import Comparison, target_confidence
from experiments.run_bilevel_trials import BilevelTrials, optimizer_observation
from experiments.run_new_target_lambda_sweep import NewTargetTrials, validate_protocol, verify_surrogate


def protocol():
    return json.loads(Path('experiments/protocols/bilevel-new-target-2000000000.json').read_text())


class NewTargetTests(unittest.TestCase):
    def test_natural_log_confidence_survives_probability_underflow(self):
        r = target_confidence(torch.tensor([-1000., 1000., 0.]), 0, 1)
        self.assertEqual(r['true_probability'], 0.)
        self.assertEqual(r['true_log_probability'], -2000.)
        self.assertEqual(r['intended_log_probability'], 0.)
        self.assertEqual(r['true_minus_intended_log_probability'], -2000.)
        self.assertTrue(math.isfinite(r['true_log_probability']))
        with self.assertRaises(FloatingPointError):
            target_confidence(torch.tensor([float('nan'), 0.]), 0, 1)

    def test_final_evaluation_reports_true_and_intended_confidence(self):
        trial = object.__new__(Comparison)
        trial.config = SimpleNamespace(precision='fp32')
        trial.test_images = torch.tensor([[math.log(2), 0., math.log(3)],
                                         [4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])
        trial.test_labels = torch.tensor([0, 0, 1, 2])
        trial.accuracy_mask = torch.tensor([False, True, True, True])
        trial.cases = [dict(key=7, target_id=0, true_label=0, intended_label=1)]
        r = trial.evaluate(torch.nn.Identity())['targets'][0]
        self.assertAlmostEqual(r['true_probability'], 2/6, places=7)
        self.assertAlmostEqual(r['intended_probability'], 1/6, places=7)
        self.assertAlmostEqual(r['true_log_probability'], math.log(2/6), places=7)
        self.assertAlmostEqual(r['intended_log_probability'], math.log(1/6), places=7)
        self.assertEqual(r['predicted_label'], 2)
        self.assertEqual(r['target_correct'], 0)
        self.assertEqual(r['attack_success'], 0)

    def test_target_observer_preserves_rng_buffers_and_mixed_modes(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                    torch.nn.Dropout(.5), torch.nn.Linear(4, 2))
        model.train()
        model[2].eval()
        trial = object.__new__(BilevelTrials)
        trial.config = SimpleNamespace(precision='fp32')
        trial.test_images = torch.rand(3, 3)
        trial.cases = [dict(key=7, target_id=0, true_label=0, intended_label=1)]
        flags = [m.training for m in model.modules()]
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        rng = torch.get_rng_state().clone()
        first = trial.observe_targets(model)
        self.assertEqual(first, trial.observe_targets(model))
        self.assertEqual(flags, [m.training for m in model.modules()])
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, buffers[k], rtol=0, atol=0)

    def test_update_observer_measures_the_applied_step_without_mutation(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        model(torch.ones(2, 3)).square().sum().backward()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        after = [p.detach().clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        r = optimizer_observation(model, optimizer, before)
        expected = sum((a.double()-b.double()).square().sum() for a, b in zip(after, before)).sqrt()
        self.assertAlmostEqual(r['parameter_update_norm'], float(expected))
        self.assertGreater(r['momentum_norm'], 0.)
        for p, value in zip(model.parameters(), after):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_predeclared_new_target_and_five_lambdas(self):
        p = protocol()
        validate_protocol(p)
        self.assertEqual(p['victim_models'], 6)
        self.assertEqual(p['victim_seeds'], [18001])
        p['lambdas'].append(.001)
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_clean_surrogate_hash_and_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'models').mkdir()
            (root/'runs').mkdir()
            (root/'models/surrogate.pt').write_bytes(b'clean weights')
            (root/'manifest.json').write_text(json.dumps(dict(config=dict(dataset='CIFAR10',
                model='ResNet18', train_size=None, epochs=40, batch_size=128,
                surrogate_seed=314159, precision='fp32'))))
            result = dict(name='surrogate', seed=314159, status='complete', poisoned=False,
                          clean_test_accuracy=.92)
            (root/'runs/surrogate.json').write_text(json.dumps(result))
            files = ['manifest.json', 'models/surrogate.pt', 'runs/surrogate.json']
            p = dict(frozen_source_files={k: hashlib.sha256((root/k).read_bytes()).hexdigest() for k in files})
            self.assertEqual(verify_surrogate(p, root)[1], result)
            (root/'models/surrogate.pt').write_bytes(b'changed weights')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                verify_surrogate(p, root)

    def test_new_poison_is_shared_by_exactly_six_fresh_victims(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = object.__new__(NewTargetTrials)
            trial.out = Path(directory)
            trial.source = trial.out/'source'
            trial.protocol = protocol()
            trial.manifest = {}
            trial.cases = [dict(key=2000000000, target_id=123, true_label=2,
                               intended_label=0, poison_ids=list(range(500)))]
            trial.full_validset = SimpleNamespace(classes=['airplane', 'bird'])
            trial.summarize_trials = Mock(return_value=dict(all_succeeded=False, attempted=6))
            trial.train_condition = Mock()
            trial.out_of_time = Mock(return_value=False)
            poison = dict(target_id=123, poison_ids=torch.arange(500), delta=torch.zeros(500, 1))
            def brew(case, state):
                path = trial.poison_path(case)
                path.parent.mkdir()
                path.write_bytes(b'new fixed poison')
                path.with_suffix('.json').write_text(json.dumps(dict(
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
                return poison['delta']
            trial.brew = Mock(side_effect=brew)
            with patch('experiments.run_new_target_lambda_sweep.load_state',
                       side_effect=lambda path: poison if path.name=='2000000000.pt' else {}):
                trial.run_new_target()
            calls = trial.train_condition.call_args_list
            self.assertEqual([c.args[2] for c in calls], [None, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5])
            self.assertEqual([c.args[1] for c in calls], [18001]*6)
            self.assertTrue(all(c.args[4] is poison['delta'] for c in calls))
            self.assertEqual(trial.brew.call_count, 1)
            status = json.loads((trial.out/'status.json').read_text())
            self.assertEqual(status['phase'], 'complete_with_failures')
            self.assertEqual(status['cached_baselines'], 0)


if __name__ == '__main__':
    unittest.main()
