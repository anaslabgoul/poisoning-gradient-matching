"""Balanced, reproducible training subsets with explicit original-row mapping."""
import copy
import numpy as np
import torch


def stratified_indices(labels, size, seed):
    labels=np.asarray(labels)
    classes=np.unique(labels)
    if size<1 or size>len(labels) or size%len(classes):
        raise ValueError('Training subset size must fit the dataset and divide evenly across classes.')
    per_class=size//len(classes)
    generator=np.random.default_rng(seed)
    selected=[]
    for label in classes:
        available=np.flatnonzero(labels==label)
        if len(available)<per_class:
            raise ValueError('Not enough examples in class '+str(label))
        selected.extend(generator.choice(available,size=per_class,replace=False).tolist())
    return np.asarray(sorted(selected),dtype=np.int64)


def reduced_dataset(dataset, size, seed):
    """Copy the dataset shell and reindex rows; avoid Subset.get_target delegation."""
    original_indices=stratified_indices(dataset.targets,size,seed)
    reduced=copy.copy(dataset)
    if isinstance(dataset.data,torch.Tensor):
        reduced.data=dataset.data[torch.as_tensor(original_indices)]
    else:
        reduced.data=dataset.data[original_indices]
    labels=np.asarray(dataset.targets)[original_indices]
    reduced.targets=(torch.as_tensor(labels) if isinstance(dataset.targets,torch.Tensor) else labels.tolist())
    return reduced,original_indices
