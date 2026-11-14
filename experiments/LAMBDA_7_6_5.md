# Fixed lambda follow-up: 1e-7, 1e-6 and 1e-5

Run exactly three new defended models on the same saved target/seed, poison,
and cached undefended baseline. No prior lambda runs, source training, poison
generation or baseline training are repeated. The frozen protocol is
`protocols/bilevel-lambda-7-6-5.json`.

Only lambda changes: full CIFAR-10, ResNet18, 40 epochs, batch 128, FP32/TF32,
learning rate 0.1, Nesterov momentum 0.9, weight decay 0.0005, the existing
scheduler and augmentation, radius 16/255, three inner ascent steps, and the
full-network per-example penalty averaged over eight sampled images. There
is no clipping, warmup, adaptive lambda, update rescaling or layer restriction.
The first 32 batches log gradient diagnostics. A numerical failure is saved
and the remaining requested values are attempted; successful prefixes
continue directly to 40 epochs. Existing checkpoints and inputs are verified
and reused across scheduler slices.

The result contains four conditions: one cached baseline and three new
defended attempts. These remain exploratory results on one known target/seed,
not independent evidence of general robustness. Failed optimization never
counts as a prevented attack.

From the isolated liparis checkout:

```bash
export PYTHONNOUSERSITE=1
ulimit -c 0
WB_REPO="$PWD" WB_ROOT="$PWD/artifacts/bilevel-lambda-7-6-5" \
WB_PYTHON=/tmp/anas-bilevel-small-lambda-20261112/venv/bin/python \
WB_RUNNER=experiments/run_lambda_7_6_5.py WB_JOB_NAME=wb-lambda-765 \
WB_HARD_STOP_UTC=2026-11-15T05:00:00+00:00 \
WB_TMUX_SOCKET=wb-lambda-765 WB_TMUX_SESSION=lambda-765 \
bash experiments/host_bilevel_trials.sh start
```

The tmux controller submits one scheduler-managed allocation at a time,
resumes exact optimizer/minibatch/RNG checkpoints between two-hour slices,
and survives laptop disconnection. Infrastructure failures stop visibly.
The cutoff is 15 November 2026, 07:00 Asia/Hebron (05:00 UTC).

Inspect `artifacts/bilevel-lambda-7-6-5/status.json`, `results.json`,
`scheduler/jobs.log`, `scheduler/slurm-<jobid>.log`, and `controller.log`.
