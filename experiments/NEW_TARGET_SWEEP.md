# New target: five lambdas and true/attacker-class confidence

Predeclare paper case key 2000000000 and retain it regardless of outcome.
Keep training seed 18001, full CIFAR-10, CIFAR-adapted ResNet18, 40 epochs,
and every previous optimizer/defense setting. Reuse the hash-verified clean
surrogate only to brew one new poison (500 images, 16/255, 250 iterations,
8 restarts). Train a fresh poisoned undefended baseline, followed by
lambda 1e-10, 1e-8, 1e-7, 1e-6 and 1e-5: six new victim models.

Confidence uses `log_softmax` of FP32 logits with FP64 accumulation and
natural logarithms. Save `true_probability`, `intended_probability`,
`true_log_probability`, `intended_log_probability` and their log-probability
difference. The false class is the attacker's predeclared intended class.
Observe initialization, early updates, checkpoints and every epoch, then
evaluate all test images at completion. Observations do not tune training.
Failed runs have no final confidence; their last finite observations are
explicitly historical. Save failure model/optimizer/RNG states separately.

Early diagnostics include the applied penalty/CE gradient ratio, actual
parameter update norm, momentum norm, per-layer total gradient norms and
per-layer BN running-stat ranges. These are read-only observations; no
clipping, warmup, normalization, adaptive weights or layer restriction.

The remote tmux controller uses scheduler allocations and exact continuation
checkpoints. Start from an isolated checkout with:

```bash
export PYTHONNOUSERSITE=1
ulimit -c 0
WB_REPO="$PWD" WB_ROOT="$PWD/artifacts/bilevel-new-target-2000000000" \
WB_PYTHON=/tmp/anas-bilevel-small-lambda-20261112/venv/bin/python \
WB_RUNNER=experiments/run_new_target_lambda_sweep.py WB_JOB_NAME=wb-new-target \
WB_HARD_STOP_UTC=2026-11-15T05:00:00+00:00 \
WB_TMUX_SOCKET=wb-new-target WB_TMUX_SESSION=new-target \
bash experiments/host_bilevel_trials.sh start
```

The cutoff is 15 November 2026, 07:00 Asia/Hebron (05:00 UTC). The controller
does not continue beyond the cutoff. Runtime depends on how many lambdas
complete rather than diverge; six attempted models are a maximum, not a
guarantee that every run finishes before the cutoff. Code is versioned;
raw research results remain in the local/remote ignored artifact directories.
