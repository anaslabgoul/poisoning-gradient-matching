"""Numerical checks of the proposed objectives; no datasets or GPU required."""

import unittest
import torch
import torch.nn.functional as F

from forest.victims.defenses import final_weight_gradients, paired_defense_loss, perturb_inputs


class DefenseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_explicit_gradient_and_derivative_match_autograd(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
        x = torch.randn(2, 4)
        y = torch.tensor([0, 2])
        h = model[:2](x)
        logits = model[2](h)
        explicit = final_weight_gradients(logits, h, y)
        reference = torch.stack([torch.autograd.grad(F.cross_entropy(logits[i:i+1], y[i:i+1]),
                                 model[2].weight, create_graph=True, retain_graph=True)[0]
                                 for i in range(2)])
        torch.testing.assert_close(explicit, reference)
        analytic_derivative = torch.autograd.grad(explicit.square().sum(), model[0].weight, retain_graph=True)[0]
        autograd_derivative = torch.autograd.grad(reference.square().sum(), model[0].weight)[0]
        torch.testing.assert_close(analytic_derivative, autograd_derivative)

    def test_perturbation_bounds_after_normalization(self):
        dm = torch.tensor([.2, .5, .7])[None, :, None, None]
        ds = torch.tensor([.1, .3, .2])[None, :, None, None]
        pixels = torch.rand(3, 3, 4, 4)
        inputs = (pixels - dm) / ds
        changed = perturb_inputs(inputs, 16, dm, ds) * ds + dm
        self.assertGreaterEqual(changed.min().item(), -1e-6)
        self.assertLessEqual(changed.max().item(), 1 + 1e-6)
        self.assertLessEqual((changed - pixels).abs().max().item(), 16 / 255 + 1e-6)

    def test_all_objectives_reduce_to_ce_at_zero_radius(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
        x, y = torch.rand(3, 4), torch.tensor([0, 1, 2])
        criterion = lambda model, outputs, labels: F.cross_entropy(outputs, labels)
        expected = criterion(model, model(x), y)
        for method in ('standard', 'random-augmentation', 'prediction-consistency', 'gradient-consistency'):
            _, loss, metrics = paired_defense_loss(model, x, y, criterion, method, radius=0)
            torch.testing.assert_close(loss, expected)
            self.assertAlmostEqual(metrics['regularizer'].item(), 0, places=6)

    def test_kl_direction_and_gradients_through_both_views(self):
        model = torch.nn.Linear(4, 3)
        x, y = torch.rand(3, 4), torch.tensor([0, 1, 2])
        criterion = lambda model, outputs, labels: F.cross_entropy(outputs, labels)
        g = torch.Generator().manual_seed(7)
        changed = perturb_inputs(x, 40, 0., 1., g)
        a, b = model(x), model(changed)
        expected = criterion(model, a, y) + 2 * (
            a.softmax(1) * (a.log_softmax(1) - b.log_softmax(1))).sum(1).mean()
        expected_grad = torch.autograd.grad(expected, model.weight)[0]
        _, actual, _ = paired_defense_loss(model, x, y, criterion, 'prediction-consistency',
                                         radius=40, weight=2, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, model.weight)[0], expected_grad)

    def test_batchnorm_updates_once_and_regularizer_reaches_features(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.BatchNorm1d(5),
                                    torch.nn.Tanh(), torch.nn.Linear(5, 3))
        x, y = torch.rand(8, 4), torch.arange(8) % 3
        criterion = lambda model, outputs, labels: F.cross_entropy(outputs, labels)
        _, loss, metrics = paired_defense_loss(model, x, y, criterion, 'gradient-consistency', radius=32)
        self.assertEqual(model[1].num_batches_tracked.item(), 1)
        self.assertGreater(metrics['regularizer'].item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model[0].weight.grad).all())
        self.assertGreater(model[0].weight.grad.abs().sum().item(), 0)

    def test_prediction_consistency_is_finite_with_underflowed_probabilities(self):
        model = torch.nn.Linear(4, 3)
        with torch.no_grad():
            model.weight[0].fill_(1000)
            model.weight[1].fill_(-1000)
            model.weight[2].zero_()
        x, y = torch.ones(4, 4)*.5, torch.tensor([0, 1, 2, 0])
        criterion = lambda model, outputs, labels: F.cross_entropy(outputs, labels)
        _, loss, _ = paired_defense_loss(model, x, y, criterion, 'prediction-consistency')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
