"""Subset correctness is essential to poison row alignment and budget counts."""
import unittest
import numpy as np
from experiments.subsets import reduced_dataset,stratified_indices


class FakeDataset:
    def __init__(self):
        self.targets=[label for label in range(3) for _ in range(10)]
        self.data=np.arange(30)[:,None]
    def get_target(self,index):
        return self.targets[index],index
    def __len__(self):
        return len(self.targets)


class TrainingSubsetTest(unittest.TestCase):
    def test_balanced_reindexed_rows_and_preserved_source(self):
        original=FakeDataset()
        reduced,ids=reduced_dataset(original,12,37)
        self.assertEqual(len(reduced),12)
        self.assertEqual(len(original),30)
        self.assertEqual(np.bincount(reduced.targets).tolist(),[4,4,4])
        self.assertEqual(len(set(ids)),12)
        for index,original_index in enumerate(ids):
            self.assertEqual(reduced.get_target(index),(original.targets[original_index],index))
            self.assertEqual(reduced.data[index,0],original_index)
        np.testing.assert_array_equal(ids,stratified_indices(original.targets,12,37))
        self.assertFalse(np.array_equal(ids,stratified_indices(original.targets,12,38)))

    def test_rejects_invalid_size_or_class_undersupply(self):
        for size in (0,11,33):
            with self.assertRaises(ValueError):
                stratified_indices(FakeDataset().targets,size,37)
        with self.assertRaises(ValueError):
            stratified_indices([0]*9+[1],6,37)


if __name__=='__main__':
    unittest.main()
