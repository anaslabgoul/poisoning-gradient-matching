# Paired-view defense comparison

The full-network min-max variant and the new three-trial, four-lambda ResNet18
study are documented separately in [BILEVEL.md](BILEVEL.md). It includes an
inner maximization and gradients of every trainable parameter; the older
`gradient-consistency` method below remains the final-layer random-view baseline.

## Evaluation across the target class

[`results/dog-group-evaluation.json`](results/dog-group-evaluation.json) records
an inference-only evaluation of the four saved victims on all 1,000 CIFAR-10
test dogs. The intended wrong class is frog, matching the existing attack.
It also records all-10,000-image test accuracy and matched clean-control
dog-to-frog confusion rates. The 999 dogs not used in poison construction
are reported separately. These data measure **class-wide transfer of the
single-image poisons**, not a poison optimization designed to attack all dogs.

Re-evaluate the saved checkpoints without any training:

```bash
python experiments/evaluate_target_group.py
```

This experiment implements the three objectives in the user's November 2026
gradient-stability proposal, alongside ordinary training. It is an initial
**fixed-poison** evaluation, not evidence of robustness to defense-aware attacks.

## Objectives

All conditions use the existing CIFAR-10 ResNet18, full training set, and repository
differentiable crop/translation/flip augmentation. For the three defenses, an
additional view is sampled by uniform pixel noise in [-r/255, r/255], clipped to
[0,1], then normalized. The same radius is used in all three conditions.

* `standard`: cross-entropy on the original augmented view.
* `random-augmentation`: half the sum of original and perturbed cross-entropies.
* `prediction-consistency`: original cross-entropy plus lambda KL(p || p').
  Gradients flow through both probability distributions (no teacher detach).
* `gradient-consistency`: original cross-entropy plus lambda times the mean
  **per-example** squared Frobenius norm of the difference in final classifier
  weight gradients. For CE and a Linear classifier these gradients are exactly
  `(softmax(logits)-onehot(label)) outer features`. Both branches and the feature
  extractor remain differentiable. Bias gradients and earlier layers are not
  part of the penalty. This is a random-direction final-layer approximation,
  not worst-case full-network gradient stability.

Paired views are concatenated into a single forward pass, so BatchNorm buffers
update once per optimizer step and original/perturbed views share batch statistics.
The standard condition uses its normal single-view batch. Gradient consistency
currently supports a single-device model. The other CLI training options retain
their previous defaults when `--defense standard` is selected.

## Running

From the repository root, with CUDA PyTorch, matching torchvision, numpy and Pillow:

```bash
python -m unittest tests.test_defenses -v
python experiments/compare_defenses.py --benchmark 16 --output artifacts/benchmark-fp32
python -u experiments/compare_defenses.py --output artifacts/defense-comparison
```

The normal `brew_poison.py` training loop also accepts `--defense`,
`--defense_radius` (pixel units), and `--defense_weight`. That script applies the
chosen training objective to **both** clean surrogate training and victim training;
it does not automatically adapt the poison objective to a defense. Use the
comparison runner to hold standard poisons fixed across defenses.

## Requested four-model comparison

The default runner now trains **four poisoned-data victim models**, one per
technique, using one case (2000000000) and one initialization (1729). A separate
clean surrogate is needed to create the poisons and calibrate the penalties;
it is cached and reused. No per-method clean controls or extra victim seeds are
run unless explicitly selected. Use `--clean-controls` to opt into clean controls.

Measured results for the first four matched models are saved in
[`results/four-model-comparison.json`](results/four-model-comparison.json).
Those saved models were evaluated on 9,997 non-target test images (the original
expanded pilot excluded three preselected targets). The attack target is scored
separately. The attack has one target image, so the per-model observed success
fraction is 0/1 or 1/1; it is not the paper's average across many target cases.
All additional experiment runs were stopped at the user's scope correction.

To run the four-model workflow from scratch:

```bash
python -u experiments/compare_defenses.py --precision bf16 --poison-batch 512 --output artifacts/four-model-comparison
```

## Expanded pilot protocol (opt-in)

The larger pilot below is **not** the default. To request it explicitly:

```bash
python -u experiments/compare_defenses.py --cases 2000000000 2100000000 2110000000 --seeds 1729 1730 --clean-controls --precision bf16 --poison-batch 512 --output artifacts/expanded-defense-comparison
```

* Full CIFAR-10, CIFAR-adapted ResNet18, 40 epochs, batch size 128.
* Repository SGD/Nesterov 0.9, weight decay 0.0005, LR 0.1, drops at 14/24/35.
* Cases 2000000000, 2100000000, 2110000000; victim seeds 1729 and 1730.
* Independent clean surrogate, seed 314159, reused for brewing all three cases.
* 1% training poison budget (500 images), epsilon 16/255, 250 iterations,
  eight restarts, single brewing model. The measured four-model comparison
  used poison batch 512 (all 500 poisons together), with effective Adam LR 0.1.
  A preliminary batch-128 optimization was discarded before victim evaluation.
* Defense radius 16/255. Each regularizer's lambda is calibrated once on the
  first 1024 **clean training** examples using the independent trained surrogate,
  to make its mean weighted penalty 10% of mean CE. Coefficients are frozen
  before victim runs. This is a scale heuristic, **not** an optimized grid search.
* Four conditions x three cases x two seeds = 24 poisoned victim trainings.
  Four conditions x two seeds = eight clean controls, plus the clean surrogate.
* Identical poisoned data, model initialization, minibatch order and common
  spatial augmentation draws for matched methods. Extra perturbation draws use
  a separate RNG so they do not change subsequent spatial augmentations/shuffles.
* Test accuracy excludes the union of target IDs in all conditions. Targeted
  attack success means the exact intended wrong class, not any misclassification.
* Report per-case averaged attack success, standard error across cases, test
  accuracy after clean and poisoned training, train time, and peak GPU allocation.

The dataset is held on GPU to avoid image decoding/network-filesystem overhead;
this preserves the normalized input tensors. FP32 is the default, with TF32 CUDA
math enabled consistently. Optional `--precision bf16` applies BF16 autocast to
all training/evaluation conditions; penalties are computed in FP32. Record any
precision change as a protocol deviation. GPU memory figures include cached data.

Outputs contain a commit/config/environment manifest, per-epoch progress, final
checkpoints, calibrated weights, poison tensors with IDs/hash/bounds, per-run raw
measurements and an aggregate `results.json`. Completed models/runs are reused on
restart; a partly completed training is restarted from its seed. Outputs with a
different commit/config are rejected to prevent mixing incompatible runs.

Do not select target cases by whether an attack succeeds, and do not select
defense weights on test accuracy or target attack outcomes. Three cases are a
small pilot: six victim outcomes per method do not establish a reliable ranking.
Run additional cases/seeds and defense-aware attacks before robustness claims.
