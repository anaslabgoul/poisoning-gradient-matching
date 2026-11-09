"""Paired-view defenses from the November 2026 gradient-stability proposal.

The radius is specified in 0--255 pixel units, independently of normalization.
Both branches remain differentiable. The gradient penalty uses per-example
gradients of the final Linear layer's *weights*, not a gradient of a batch mean.
"""

import torch
import torch.nn.functional as F


METHODS = ('standard', 'random-augmentation', 'prediction-consistency', 'gradient-consistency')


def perturb_inputs(inputs, radius, dm, ds, generator=None):
    """Uniform pixel-space jitter, clipped to the valid image range."""
    if radius < 0:
        raise ValueError('Defense radius must be nonnegative.')
    noise = torch.rand(inputs.shape, device=inputs.device, dtype=inputs.dtype,
                       generator=generator) * 2 - 1
    pixels = inputs * ds + dm
    pixels = (pixels + noise * (radius / 255)).clamp(0, 1)
    return (pixels - dm) / ds


def final_weight_gradients(logits, features, labels):
    """Exact unaveraged CE weight gradients, retaining their derivative graph."""
    residual = logits.float().softmax(dim=1) - F.one_hot(labels, logits.shape[1]).float()
    return residual.unsqueeze(2) * features.float().unsqueeze(1)


def paired_defense_loss(model, inputs, labels, loss_fn, method='standard',
                       radius=16., weight=1., dm=0., ds=1., generator=None):
    """Return original-view logits, the training loss, and detached diagnostics.

    Nonstandard methods use one concatenated forward pass, so BatchNorm running
    statistics advance exactly once per optimizer step in every condition.
    Their normalization statistics are shared between original and jittered views.
    """
    if method not in METHODS:
        raise ValueError('Unknown defense: ' + method)
    if weight < 0:
        raise ValueError('Defense weight must be nonnegative.')
    if method == 'standard':
        outputs = model(inputs)
        loss = loss_fn(model, outputs, labels)
        return outputs, loss, {'classification': loss.detach(), 'regularizer': loss.detach() * 0}

    perturbed = perturb_inputs(inputs, radius, dm, ds, generator)
    captured = []
    handle = None
    if method == 'gradient-consistency':
        if isinstance(model, torch.nn.DataParallel):
            raise ValueError('Gradient consistency currently supports a single-device model.')
        linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        if not linear_layers:
            raise ValueError('Gradient consistency requires a final Linear classifier.')
        handle = linear_layers[-1].register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
    try:
        both_outputs = model(torch.cat((inputs, perturbed), dim=0))
    finally:
        if handle is not None:
            handle.remove()
    outputs, perturbed_outputs = both_outputs.chunk(2)
    classification = loss_fn(model, outputs, labels)
    if method == 'random-augmentation':
        loss = (classification + loss_fn(model, perturbed_outputs, labels)) / 2
        regularizer = classification.detach() * 0
    elif method == 'prediction-consistency':
        # KL(p(original) || p(perturbed)), with gradients through both arguments.
        # log_target avoids log(0) in the target derivative when softmax
        # underflows; unlike a detached teacher, both branches still get gradients.
        regularizer = F.kl_div(perturbed_outputs.float().log_softmax(dim=1),
                              outputs.float().log_softmax(dim=1),
                              log_target=True, reduction='batchmean')
        loss = classification + weight * regularizer
    else:
        if len(captured) != 1 or captured[0].shape[0] != both_outputs.shape[0]:
            raise ValueError('Expected one final classifier invocation for the paired batch.')
        features, perturbed_features = captured[0].chunk(2)
        gradients = final_weight_gradients(outputs, features, labels)
        perturbed_gradients = final_weight_gradients(perturbed_outputs, perturbed_features, labels)
        regularizer = (gradients - perturbed_gradients).square().sum(dim=(1, 2)).mean()
        loss = classification + weight * regularizer
    return outputs, loss, {'classification': classification.detach(),
                           'regularizer': regularizer.detach()}
