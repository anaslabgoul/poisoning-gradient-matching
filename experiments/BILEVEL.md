# Full-network min-max gradient consistency: three matched trials

This implements the user's clarified scope: **three independent target/seed
trials**, each comparing ordinary training against lambda **1, 0.1, 0.001,
0.0001**, for **15 victim models**, plus one independent clean brewing surrogate.
It does not launch the previously proposed three separate ablation/adaptive studies.

## Frozen experiment

`protocols/bilevel-resnet18-three-trials.json` is the complete specification.
Use full CIFAR-10 (50,000 training images), the original repository's
CIFAR-adapted ResNet18, 40 epochs and batch size 128. Repository SGD uses LR
0.1, Nesterov momentum 0.9 and weight decay 0.0005, with its existing scheduler.
Targets/poison subsets are drawn from three newly declared case keys before
outcomes. Victim seeds are independent of the clean source. Targets and poison
row IDs are saved in the manifest; retain every trial regardless of attack success.

Witches' Brew gradient matching: 1% budget (500 clean-label images), pixel
epsilon 16/255, 250 iterations, eight restarts, one independent clean source,
poison batch 128. Each trial reuses the exact saved poison across its five victims.
Initialization, shuffling and spatial augmentation draws match within each trial;
the defense uses a separate random generator. This is a fixed-poison evaluation,
**not an adaptive attack evaluation or a reproduction of all paper results**.

## Objective and derivatives

For each image, g(x,y) is the gradient of its cross-entropy with respect to
**all 11,173,962 trainable parameters** of ResNet18: convolutions, classifier,
biases and BatchNorm affine parameters. The regularizer is the sum of squared
differences across these parameters, averaged across examples. There is no
parameter-count normalization, layer selection, subsampling, or detached teacher.

Hold theta fixed and approximately maximize ||g(x+delta,y)-g(x,y)||^2 over the
pixel L-infinity ball of radius 16/255 intersected with [0,1]. Use a feasible
random start, three projected sign-ascent steps (step size 32/3 pixel units),
and retain each example's best perturbation across the entire trajectory.
The maximum is approximate/local, not a claimed global maximum.

Then update theta on CE(x,y) + lambda * mean_i regularizer_i. Treat the selected
perturbation as fixed during this outer step (a Danskin-style min-max update);
do not unroll the optimizer. Both outer gradient branches remain differentiable.
The exact per-example calculation is vectorized in chunks of 16 and accumulated
before one optimizer update. Chunking reduces memory without changing the norm.

To make the per-example gradients well-defined and avoid changing running
statistics during ascent, **penalty branches use frozen BN running statistics
and disabled dropout**. The ordinary classification forward uses training mode
and updates the BN buffers once. BN weights/biases remain trainable and included.
FP32 tensors and derivatives are used throughout, with TF32 enabled for all
conditions. The implementation checks finite losses and derivatives. A numerical
failure is reported as a failure, never interpreted as a successfully blocked
attack, and never repaired by silently changing lambda or clipping gradients.

## Running and surviving disconnection

Checks:

```bash
python -m unittest tests.test_bilevel tests.test_bilevel_runner -v
python experiments/benchmark_bilevel.py --batch 128 --chunk 16
```

Start on the host **from the isolated checkout containing the committed code**:

```bash
WB_REPO="$PWD" bash experiments/host_bilevel_trials.sh start
```

The `wb-bilevel` tmux server, session `resnet18`, hosts a persistent controller.
It submits exactly one exclusive, single-task, two-hour Slurm allocation at a
time on liparis. Each allocation runs Python via srun; no training escapes the
scheduler. The runner saves and yields after a 100-minute slice. The controller
continues only after the explicit checkpointed/exit-75 state; unexpected
infrastructure failures stop visibly rather than repeatedly launching jobs.
Neither the laptop nor its SSH connection needs to remain on.

Checkpoints restore model and BN state, optimizer momentum, scheduler, epoch,
the current permutation and next minibatch, all common and defense RNGs, raw
epoch statistics and timing. Atomic checkpoints are written every two minutes,
at epoch boundaries and before yielding. A crash resumes from the latest saved
boundary; completed models and saved poisons are reused. The immutable config
and execution commit prevent accidental mixing of runs. No existing experiment
or stopped grid is resumed, overwritten or cancelled.

Check progress:

```bash
cat artifacts/bilevel-resnet18/status.json
cat artifacts/bilevel-resnet18/results.json
tail -30 artifacts/bilevel-resnet18/controller.log
cat artifacts/bilevel-resnet18/scheduler/jobs.log
tmux -L wb-bilevel attach -t resnet18
```

## Reporting

Save all raw target predictions/probabilities, exact intended-label attack success,
test accuracy on the same 9,997 non-target images, full test accuracy, training
time, peak allocation, per-epoch CE and penalty, initial/maximized inner values,
poison IDs/bounds/hashes, final checkpoints and any numerical failures. Report
per-trial paired changes and descriptive means; three trials are preliminary and
do not support precise population attack-success rates or significance claims.
No per-lambda unpoisoned victim controls are included in the authorized 15-model
scope, so this study cannot isolate the clean-data accuracy cost of regularization.
