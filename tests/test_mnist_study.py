import unittest
import torch

from forest.victims.models import get_model
from forest.victims.defenses import METHODS, paired_defense_loss


class MNISTStudyTest(unittest.TestCase):
    def test_uncertainty_for_zero_or_all_successes(self):
        from experiments.overnight_mnist import wilson,mean_ci
        low,high=wilson(0,25)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,.13)
        self.assertLess(high,.14)
        low,high=wilson(25,25)
        self.assertGreater(low,.86)
        self.assertAlmostEqual(high,1.)
        self.assertEqual(mean_ci([.9]*25),[.9,.9])

    def test_small_model_and_all_defense_gradients(self):
        torch.set_num_threads(2)
        model=get_model('LeNet','MNIST')
        self.assertEqual(sum(p.numel() for p in model.parameters()),61706)
        inputs=torch.rand(4,1,28,28)
        labels=torch.arange(4)
        criterion=lambda model,outputs,labels:torch.nn.functional.cross_entropy(outputs,labels)
        for method in METHODS:
            model.zero_grad(set_to_none=True)
            outputs,loss,_=paired_defense_loss(model,inputs,labels,criterion,method,radius=32)
            self.assertEqual(tuple(outputs.shape),(4,10))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
