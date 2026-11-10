"""The grid must vary poisoning, not silently change the target or reuse noise."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch

from experiments.compare_defenses import Comparison
from experiments.fixed_target_grid import define_grid,index_hash


class FixedTargetGridTest(unittest.TestCase):
    def test_ten_targets_ten_seeds_and_fresh_poison_subsets(self):
        labels=np.repeat(np.arange(10),1000)
        targets,seeds,runs,ids=define_grid(labels,labels,[str(i) for i in range(10)],excluded=[0,1000])
        self.assertEqual(seeds,list(range(11001,11011)))
        self.assertEqual(len({t['target_id'] for t in targets}),10)
        self.assertEqual(len(runs),100)
        self.assertEqual(len({r['key'] for r in runs}),100)
        self.assertEqual(ids.shape,(100,400))
        for index,target in enumerate(targets):
            rows=runs[index*10:(index+1)*10]
            self.assertEqual({r['target_id'] for r in rows},{target['target_id']})
            self.assertNotEqual(target['true_label'],target['intended_label'])
            self.assertEqual(len({index_hash(np.sort(ids[r['run_index']])) for r in rows}),10)
            for row in rows:
                selected=ids[row['run_index']]
                self.assertEqual(len(set(selected)),400)
                self.assertTrue((labels[selected]==target['intended_label']).all())
                self.assertEqual(row['poison_ids_sha256'],index_hash(selected))

    def test_poison_seed_is_applied_after_surrogate_rng_reset(self):
        with TemporaryDirectory() as directory:
            results=[]
            for seed in (1011001,1011002):
                experiment=Comparison.__new__(Comparison)
                experiment.out=Path(directory)/str(seed)
                experiment.device=torch.device('cpu')
                experiment.setup={'device':torch.device('cpu'),'dtype':torch.float32}
                experiment.args=SimpleNamespace(budget=.04,pbatch=2)
                experiment.config=SimpleNamespace(fixed_cases=True,eps=32.)
                experiment.images=torch.zeros(3,1,2,2)
                experiment.labels=torch.tensor([0,1,1])
                experiment.ds=torch.ones(1,1,1,1)
                dataset=[(torch.zeros(1,2,2),i%2,i) for i in range(3)]
                experiment.full_validset=dataset
                experiment.kettle=SimpleNamespace(trainset=dataset,random_construction=Mock())
                case=dict(key=4100000000,target_id=0,true_label=0,intended_label=1,
                          poison_ids=[1,2],poison_seed=seed)
                def make_victim(*args,**kwargs):
                    torch.manual_seed(314159)
                    return SimpleNamespace(model=Mock())
                witch=SimpleNamespace(stat_optimal_loss=0.,brew=lambda victim,kettle:torch.rand(2,1,2,2)*.01)
                with patch('experiments.compare_defenses.forest.Victim',side_effect=make_victim), \
                     patch('experiments.compare_defenses.forest.Witch',return_value=witch), \
                     patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache'), \
                     patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.max_memory_allocated',return_value=0):
                    delta=experiment.brew(case,{})
                torch.manual_seed(seed)
                torch.testing.assert_close(delta,torch.rand(2,1,2,2)*.01)
                self.assertEqual(experiment.kettle.target_ids.tolist(),[0])
                experiment.kettle.random_construction.assert_not_called()
                results.append(delta)
            self.assertFalse(torch.equal(results[0],results[1]))


if __name__=='__main__':
    unittest.main()
