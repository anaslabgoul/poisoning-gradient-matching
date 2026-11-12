"""Exact per-example, full-network gradient-consistency min-max training.

The norm is a SUM over every trainable parameter (including biases and BN
affine parameters), then a MEAN over examples. No parameter subsampling,
last-layer approximation, finite differences, or parameter-count normalization.
The inner solver is projected ascent, not a guaranteed global maximizer.
"""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


@contextmanager
def fixed_statistics(model):
    """Define per-example gradients with fixed BN statistics and no dropout.

Restore every module's flag, including deliberately mixed train/eval states.
The ordinary classification forward alone updates BN running statistics.
"""
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in flags:
            module.training = training


def example_gradients(model, inputs, labels):
    """Differentiable CE gradients for ALL trainable network parameters."""
    parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    if not parameters:
        raise ValueError('Gradient consistency needs trainable parameters.')
    constants = {k: p for k, p in model.named_parameters() if not p.requires_grad}
    constants.update(dict(model.named_buffers()))

    def loss(params, image, label):
        logits = functional_call(model, (params, constants), (image.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))

    with fixed_statistics(model):
        return vmap(grad(loss), in_dims=(None, 0, 0), randomness='error')(
            parameters, inputs, labels)


def squared_difference(first, second):
    return sum((first[key] - second[key]).flatten(1).square().sum(1) for key in first)


def gradient_penalties(model, inputs, labels, pixel_delta, dm=0., ds=1.):
    """One full-network squared gradient difference per example."""
    changed = (inputs * ds + dm + pixel_delta).clamp(0, 1)
    changed = (changed - dm) / ds
    return squared_difference(example_gradients(model, inputs, labels),
                              example_gradients(model, changed, labels))


def maximize_gradient_difference(model, inputs, labels, radius=16., steps=3,
                                 step_size=None, dm=0., ds=1., generator=None,
                                 initial_delta=None):
    """Best-of-trajectory per-example projected sign ascent in pixel space.

Model parameters remain fixed. Autograd computes mixed input/parameter
derivatives without writing to parameter .grad fields. A nonzero random start
avoids the stationary point at delta=0. Return a detached perturbation: outer
training uses a Danskin-style update, without unrolling the ascent trajectory.
"""
    if not math.isfinite(radius) or radius < 0 or steps < 0:
        raise ValueError('Require a finite nonnegative radius and nonnegative steps.')
    if step_size is None:
        step_size = radius / max(steps, 1) * 2
    if not math.isfinite(step_size) or step_size < 0:
        raise ValueError('Step size must be finite and nonnegative (pixel units).')
    pixels = (inputs.detach() * ds + dm).clamp(0, 1)
    lower = torch.maximum(torch.full_like(pixels, -radius / 255), -pixels)
    upper = torch.minimum(torch.full_like(pixels, radius / 255), 1 - pixels)
    if initial_delta is None:
        delta = torch.rand(inputs.shape, device=inputs.device, dtype=inputs.dtype,
                           generator=generator) * (2 * radius / 255) - radius / 255
    else:
        delta = initial_delta.detach().clone()
    delta = torch.maximum(torch.minimum(delta, upper), lower).detach()
    original = {key: value.detach() for key, value in example_gradients(model, inputs.detach(), labels).items()}
    best_delta = delta.clone()
    best_score = inputs.new_full((len(inputs),), -float('inf'))
    initial_score = None
    for iteration in range(steps + 1):
        delta.requires_grad_(iteration < steps)
        changed = (pixels + delta - dm) / ds
        scores = squared_difference(original, example_gradients(model, changed, labels))
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite full-network inner penalty.')
        if initial_score is None:
            initial_score = scores.detach().clone()
        improve = scores.detach() > best_score
        best_delta = torch.where(improve.reshape(-1, *([1] * (inputs.ndim - 1))),
                                 delta.detach(), best_delta)
        best_score = torch.maximum(best_score, scores.detach())
        if iteration < steps:
            direction = torch.autograd.grad(scores.sum(), delta)[0]
            if not torch.isfinite(direction).all():
                raise FloatingPointError('Nonfinite inner ascent derivative.')
            delta = delta.detach() + step_size / 255 * direction.sign()
            delta = torch.maximum(torch.minimum(delta, upper), lower).detach()
    return best_delta.detach(), dict(initial=initial_score, maximized=best_score)


def bilevel_backward(model, inputs, labels, weight, radius=16., steps=3,
                     chunk_size=2, dm=0., ds=1., generator=None, step_size=None,
                     classification_bf16=False, regularizer_samples=None,
                     gradient_diagnostics=False):
    """Accumulate exact outer CE + lambda * mean(full-gradient penalty) grads.

Chunking changes memory use only, not the per-example objective. Classification
uses the ordinary training-mode batch; penalty branches use fixed BN buffers.
    Caller zeros gradients and steps the optimizer. Penalties use FP32. Optional
    BF16 classification preserves the previous benchmark's baseline precision.
    An optional uniform subset of images provides an unbiased Monte Carlo
    estimate of the batch's per-example penalty; it never subsets parameters.
"""
    if not math.isfinite(weight) or weight < 0 or chunk_size < 1:
        raise ValueError('Require a finite nonnegative lambda and positive chunk size.')
    if inputs.dtype != torch.float32:
        raise ValueError('Full-network training requires FP32 inputs.')
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16,
                        enabled=classification_bf16):
        logits = model(inputs)
        classification = F.cross_entropy(logits.float(), labels)
    if not torch.isfinite(classification):
        raise FloatingPointError('Nonfinite classification loss.')
    classification.backward()
    parameters = tuple(p for p in model.parameters() if p.grad is not None) if gradient_diagnostics else ()
    ce_gradients = [p.grad.detach().clone() for p in parameters]
    total, initial, maximized = inputs.new_zeros(()), inputs.new_zeros(()), inputs.new_zeros(())
    if weight and radius:
        if regularizer_samples is not None:
            if regularizer_samples < 1:
                raise ValueError('Regularizer sample count must be positive.')
            selected = torch.randperm(len(inputs), device=inputs.device, generator=generator)[:regularizer_samples]
            penalty_inputs, penalty_labels = inputs[selected], labels[selected]
        else:
            penalty_inputs, penalty_labels = inputs, labels
        count = len(penalty_inputs)
        for start in range(0, count, chunk_size):
            x, y = penalty_inputs[start:start+chunk_size], penalty_labels[start:start+chunk_size]
            delta, diagnostics = maximize_gradient_difference(
                model, x, y, radius, steps, step_size, dm, ds, generator)
            penalties = gradient_penalties(model, x, y, delta, dm, ds)
            if not torch.isfinite(penalties).all():
                raise FloatingPointError('Nonfinite outer full-network penalty.')
            (weight * penalties.sum() / count).backward()
            total += penalties.detach().sum() / count
            initial += diagnostics['initial'].sum() / count
            maximized += diagnostics['maximized'].sum() / count
    metrics = dict(classification=classification.detach(), regularizer=total,
                   loss=classification.detach() + weight * total,
                   inner_initial=initial, inner_maximized=maximized)
    if gradient_diagnostics:
        penalty_gradients = [p.grad.detach()-g for p,g in zip(parameters,ce_gradients)]
        ce_squared = sum(g.double().square().sum() for g in ce_gradients)
        penalty_squared = sum(g.double().square().sum() for g in penalty_gradients)
        dot = sum((a.double()*b.double()).sum() for a,b in zip(ce_gradients,penalty_gradients))
        metrics.update(ce_gradient_norm=ce_squared.sqrt(),
            weighted_regularizer_gradient_norm=penalty_squared.sqrt(),
            regularizer_to_ce_gradient_ratio=(penalty_squared/(ce_squared+1e-60)).sqrt(),
            gradient_cosine=dot/((ce_squared*penalty_squared).sqrt()+1e-30))
    return logits.detach(), metrics
