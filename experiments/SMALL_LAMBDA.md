# Two smaller fixed lambdas on liparis

Run exactly two new defended models: **lambda 1e-8 and 1e-10**. Reuse the
saved full-CIFAR10/ResNet18 poison and completed undefended model from the
single-case overnight study. No additional targets, source models, attacks,
or undefended training are generated.

All mathematical training settings stay unchanged: 40 epochs, batch 128,
SGD learning rate 0.1, Nesterov momentum 0.9, weight decay 0.0005, original
schedule and augmentation, FP32 with TF32 enabled, radius 16/255, three inner
ascent steps and the mean exact full-network penalty over eight uniformly
sampled images per classification batch. There is no clipping, warmup,
adaptive lambda, update rescaling, layer restriction or penalty normalization.
The frozen specification is `protocols/bilevel-small-lambda.json`.

Before any training, verify the original manifest, poison, baseline record,
and baseline checkpoint hashes. Keep the same target, labels, poison IDs,
initialization seed, minibatch order and spatial augmentation draws. This is
an exploratory comparison on a previously observed case, not fresh validation.

The first 32 batches of each condition emit detailed gradient diagnostics.
A successful prefix continues directly to 40 epochs with the same model and
optimizer state. A numerical failure is recorded and the other requested
lambda is attempted. No automatic parameter changes or extra lambda values
are permitted. Check finite BN buffers as well as weights and gradients.
Diagnostic norms use FP64 accumulation; the reported regularizer gradient
contribution is the change in accumulated FP32 gradients after adding the
weighted penalty, so very small contributions can be affected by rounding.
The diagnostics are observational and do not change gradients or RNG draws.

The host's previous PyTorch installation was incomplete. A separate runtime
uses the same PyTorch 2.8.0+cu128 and torchvision 0.23.0. Runtime/cache files
are held in a private directory on liparis's local disk because the initial
download exceeded the account's home-directory quota. Saved study files and
checkpoints remain in the isolated project checkout.

Start from the isolated checkout:

```bash
export PYTHONNOUSERSITE=1
WB_REPO="$PWD" WB_ROOT="$PWD/artifacts/bilevel-small-lambda" \
WB_PYTHON=/tmp/anas-bilevel-small-lambda-20261112/venv/bin/python \
WB_RUNNER=experiments/run_small_lambda.py WB_JOB_NAME=wb-small-lambda \
WB_HARD_STOP_UTC=2026-11-13T05:00:00+00:00 \
WB_TMUX_SOCKET=wb-small-lambda WB_TMUX_SESSION=small-lambda \
bash experiments/host_bilevel_trials.sh start
```

The remote tmux controller submits one scheduler-managed two-hour slice at
a time and resumes exact minibatch/optimizer/RNG checkpoints. It survives
laptop shutdown. Only verified checkpoint yields continue automatically;
infrastructure failures stop visibly. The fixed cutoff is 13 November 2026
at 07:00 Asia/Hebron (05:00 UTC), with no following-day continuation.

Inspect `artifacts/bilevel-small-lambda/status.json`, `results.json`,
`scheduler/jobs.log`, `scheduler/slurm-<jobid>.log`, and `controller.log`.
The summary contains one cached baseline and two newly attempted conditions.
Report each raw target outcome, ordinary test accuracy, raw/weighted penalty,
gradient norms/alignment, timing, memory and failures. A failed optimizer
never counts as a successfully prevented attack; a tiny finite penalty may
also have negligible effect and therefore needs accuracy/attack evaluation.
