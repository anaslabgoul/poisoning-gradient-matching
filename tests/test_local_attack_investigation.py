"""Verify the exact gradient aggregation used to diagnose transfer failure."""
import unittest
import torch
import torch.nn.functional as F
from experiments.investigate_local_attack import gradient_vector


class LocalAttackInvestigationTest(unittest.TestCase):
    def test_chunked_mean_matches_full_gradient_with_unequal_chunks(self):
        torch.manual_seed(19)
        model=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Tanh(),torch.nn.Linear(7,3))
        inputs=torch.randn(11,4)
        labels=torch.arange(11)%3
        expected=torch.cat([g.flatten() for g in torch.autograd.grad(
            F.cross_entropy(model(inputs),labels),model.parameters())])
        for chunk in (1,4,11):
            actual,_=gradient_vector(model,inputs,labels,batch=chunk)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-7)


if __name__=='__main__':
    unittest.main()
