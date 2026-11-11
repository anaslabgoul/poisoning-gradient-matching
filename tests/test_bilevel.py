"""Mathematical and state-isolation checks for full-network min-max training."""
import copy
import unittest

import torch
import torch.nn.functional as F

from forest.victims.bilevel import (example_gradients, gradient_penalties,
                                    maximize_gradient_difference, bilevel_backward)


class BilevelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
        self.x, self.y = torch.rand(3, 3), torch.tensor([0, 1, 0])

    def test_every_parameter_and_outer_derivative_match_explicit_autograd(self):
        delta = torch.full_like(self.x, .02)
        actual = gradient_penalties(self.model, self.x, self.y, delta)
        expected = []
        for x, y, d in zip(self.x, self.y, delta):
            a = torch.autograd.grad(F.cross_entropy(self.model(x[None]), y[None]),
                                    tuple(self.model.parameters()), create_graph=True)
            b = torch.autograd.grad(F.cross_entropy(self.model((x+d)[None]), y[None]),
                                    tuple(self.model.parameters()), create_graph=True)
            expected.append(sum((u-v).square().sum() for u, v in zip(a, b)))
        expected = torch.stack(expected)
        self.assertEqual(set(example_gradients(self.model, self.x, self.y)), set(dict(self.model.named_parameters())))
        torch.testing.assert_close(actual, expected)
        first = torch.autograd.grad(actual.mean(), tuple(self.model.parameters()), retain_graph=True)
        second = torch.autograd.grad(expected.mean(), tuple(self.model.parameters()))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_inner_solver_bounds_improvement_and_no_parameter_mutation(self):
        before = copy.deepcopy(self.model.state_dict())
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        x = self.x.clone()
        x[0, 0], x[1, 0] = 0., 1.
        dm, ds = .2, .3
        delta, scores = maximize_gradient_difference(self.model, (x-dm)/ds, self.y,
            radius=16, steps=3, dm=dm, ds=ds, generator=torch.Generator().manual_seed(23))
        self.assertFalse(delta.requires_grad)
        self.assertLessEqual(float(delta.abs().max()), 16/255+1e-7)
        self.assertGreaterEqual(float((x+delta).min()), 0.)
        self.assertLessEqual(float((x+delta).max()), 1.)
        self.assertTrue(bool((scores['maximized'] >= scores['initial']).all()))
        self.assertGreater(float((scores['maximized']-scores['initial']).sum()), 0.)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        for p in self.model.parameters():
            torch.testing.assert_close(p.grad, torch.ones_like(p))

    def test_chunking_preserves_exact_objective_and_outer_gradient(self):
        models = [copy.deepcopy(self.model), copy.deepcopy(self.model)]
        metrics = []
        for model, chunk in zip(models, (1, 3)):
            # At zero ascent steps the common deterministic initial delta is
            # easier to compare by a direct fixed-delta accumulated reference.
            F.cross_entropy(model(self.x), self.y).backward()
            delta = torch.full_like(self.x, .02)
            score = 0.
            for start in range(0, len(self.x), chunk):
                penalty = gradient_penalties(model, self.x[start:start+chunk], self.y[start:start+chunk], delta[start:start+chunk])
                (.1*penalty.sum()/len(self.x)).backward()
                score += float(penalty.detach().sum()/len(self.x))
            metrics.append(score)
        self.assertAlmostEqual(*metrics, places=6)
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_bn_updates_once_and_penalty_reaches_early_layers(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                     torch.nn.Tanh(), torch.nn.Linear(4, 2))
        model.train()
        _, metrics = bilevel_backward(model, self.x, self.y, .01, steps=1, chunk_size=2)
        self.assertEqual(int(model[1].num_batches_tracked), 1)
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertGreater(float(metrics['regularizer']), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_lambda_matches_ordinary_training(self):
        other = copy.deepcopy(self.model)
        _, metrics = bilevel_backward(self.model, self.x, self.y, 0.)
        expected = F.cross_entropy(other(self.x), self.y)
        expected.backward()
        torch.testing.assert_close(metrics['loss'], expected.detach())
        for a, b in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad)


if __name__ == '__main__':
    unittest.main()
