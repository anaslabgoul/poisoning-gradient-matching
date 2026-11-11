# One overnight trial: full CIFAR-10 and paper ResNet18

This supersedes the cancelled three-trial/four-lambda long experiment. It runs
**one attack trial**, comparing **one undefended victim and one defended victim
at lambda 0.001**, plus one independent clean brewing source. No lambda sweep,
extra target cases, or adaptive attacks are launched.

Use the original repository's CIFAR-adapted ResNet18 (11,173,962 parameters),
full 50,000-image CIFAR-10 training set, 40 epochs, batch 128, original SGD and
augmentation setup. Witches' Brew attack: 500 clean-label poison images (1%),
16/255 pixel bound, 250 brewing iterations, eight restarts, poison batch 128.
The two victims share the exact saved poison, initialization seed, shuffles and
spatial augmentations. The single predeclared target is retained even if the
baseline attack fails. This matches the paper's dataset/model/threat setting,
not its complete multi-case numerical evaluation.

## Overnight estimator

Classification uses **all 128 training images** in every batch. Uniformly
sample **eight images without replacement** from the batch, redraw them on every
step using a separate defense RNG, and compute their exact per-example
cross-entropy gradients with respect to **every trainable network parameter**.
Include convolutional weights, classifier weights/biases and BN affine
parameters. Average the eight squared full-network gradient differences; do
not divide by 128 or by the parameter count. This is a Monte Carlo estimate of
the per-example batch average, not a last-layer approximation or a reduced
training dataset. It increases regularizer estimator variance and does not
reproduce the full-128-image penalty updates used by the cancelled experiment.

For each sampled image, start with feasible random pixel noise and perform
three projected sign-ascent steps maximizing gradient disagreement, within
16/255 and [0,1]. Keep the best perturbation per image. Then optimize the
full-batch classification loss plus 0.001 times the sampled penalty mean,
holding perturbations fixed while differentiating both gradient branches.
The inner maximizer is approximate/local, not guaranteed global. Penalties
use fixed BN running statistics and no dropout; the classification forward
alone updates BN buffers. All tensors/derivatives are FP32; TF32 is enabled
consistently. No layers or parameter coordinates are sampled.

The final 128-image ResNet18 GPU benchmark with eight penalty examples measured
0.325 seconds per warm update on liparis, projecting about 1.4 hours for the
defended training. Including source training, brewing, baseline, checkpoint
overhead and queueing, this is an estimate, not a completion guarantee.

## One-night cutoff and persistent execution

The frozen protocol is `protocols/bilevel-resnet18-overnight-once.json`.
It stops no later than **12 November 2026, 07:00 Asia/Hebron (05:00 UTC)**,
checkpointing before that time if incomplete. It does not resume the following
day. Checkpoint slices still obey the host's two-hour allocation limit.
An intentional slice yield has a job-specific exit receipt; only verified
checkpointed slices continue. Numerical and infrastructure failures are visible.

From the isolated checkout on the remote host:

```bash
WB_REPO="$PWD" WB_ROOT="$PWD/artifacts/bilevel-overnight" \
WB_RUNNER=experiments/run_bilevel_overnight.py WB_JOB_NAME=wb-bilevel-night \
WB_HARD_STOP_UTC=2026-11-12T05:00:00+00:00 \
WB_TMUX_SOCKET=wb-bilevel-night WB_TMUX_SESSION=overnight \
bash experiments/host_bilevel_trials.sh start
```

The tmux controller and scheduler worker survive laptop shutdown/disconnection.
The earlier `wb-bilevel` controller and ResNet18 job 4621 were stopped, with no
old jobs or workers remaining. Nothing from the cancelled run is resumed.

Inspect `artifacts/bilevel-overnight/status.json`, `results.json`, `manifest.json`,
`scheduler/jobs.log` and `controller.log`. Save target predictions/probabilities,
intended-class success for each model, matched non-target test accuracy, full
test accuracy, penalties/ascent progress, time/memory and all artifacts. One
trial is a proof of concept, not an estimate of population attack success or
statistically established robustness.
