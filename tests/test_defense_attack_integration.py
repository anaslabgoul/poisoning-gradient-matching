"""CUDA integration checks on synthetic images, without downloading datasets."""

from types import SimpleNamespace
import unittest

import torch

import forest
from forest.data.diff_data_augmentation import RandomTransform
from forest.victims.defenses import METHODS, paired_defense_loss
from forest.victims.models import get_model


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class DefenseAttackIntegration(unittest.TestCase):
    def test_real_resnet_training_objectives_and_brewing(self):
        torch.set_num_threads(4)
        torch.manual_seed(123)
        device = torch.device('cuda:0')
        setup = dict(device=device, dtype=torch.float32)
        inputs = torch.rand(8, 3, 32, 32, device=device)
        labels = torch.ones(8, device=device, dtype=torch.long)
        criterion = lambda model, outputs, labels: torch.nn.functional.cross_entropy(outputs, labels)
        model = get_model('ResNet18', 'CIFAR10').to(device)
        for method in METHODS:
            model.zero_grad(set_to_none=True)
            _, loss, _ = paired_defense_loss(model, inputs, labels, criterion, method,
                                             radius=16, weight=.01)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.conv1.weight.grad.abs().sum().item(), 0)
        del model
        args = forest.options().parse_args(['--attackiter', '5', '--restarts', '1', '--pbatch', '8'])
        victim = forest.Victim(args, setup=setup)
        ids = torch.arange(8)
        kettle = SimpleNamespace(
            poisonset=list(range(8)), targetset=[(inputs[0], 0, 0)],
            poison_setup={'intended_class': [2]}, dm=torch.zeros(1, 3, 1, 1, device=device),
            ds=torch.ones(1, 3, 1, 1, device=device), poison_lookup=dict(zip(range(8), range(8))),
            poisonloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, labels, ids),
                                                    batch_size=8, num_workers=0),
            augment=RandomTransform(32, 32, shift=8, fliplr=True),
            initialize_poison=lambda: torch.zeros(8, 3, 32, 32))
        witch = forest.Witch(args, setup=setup)
        delta = witch.brew(victim, kettle)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(delta.abs().max().item(), args.eps/255+1e-6)
        poisoned = inputs.cpu()+delta
        self.assertGreaterEqual(poisoned.min().item(), -1e-6)
        self.assertLessEqual(poisoned.max().item(), 1+1e-6)


if __name__ == '__main__':
    unittest.main()
