# MNIST / LeNet overnight statistics

Run in remote tmux; no local computer or SSH connection needs to stay on:

```bash
cd ~/poisoning-gradient-matching
tmux new-session -d -s mnist-defense-25 'cd ~/poisoning-gradient-matching && ~/.venvs/witches-brew/bin/python -u experiments/overnight_mnist.py > artifacts/mnist-lenet-25-study.log 2>&1'
```

Default study: MNIST 28x28 grayscale, LeNet (61,706 parameters), full 60,000
training samples, 20 epochs, batch 128, repository SGD/Nesterov and LR drops.
Each of 25 predeclared independent cases selects a target digit, wrong label,
poison subset and fresh victim seed. Four methods share exactly the same poison,
initial weights, batch order, and spatial augmentations within each case:
standard training, random perturbation augmentation, prediction consistency,
and final-layer gradient consistency. Thus 25 victim trainings per method,
100 reported victim trainings total. One clean surrogate and three to nine
development victim trainings are additional preparation, excluded from statistics.

Digit augmentation uses translations, never horizontal/vertical flips. FP32
training/penalties, TF32 CUDA enabled. Defense perturbation radius 32/255.
Weights are calibrated once on clean training data, not test attack outcomes.

Fixed, non-adaptive gradient-matching attack: epsilon 32/255, 250 iterations,
eight restarts, poison batch 512. A development-only grid tries 1% cosine, then
4% cosine, then 4% squared-gradient matching (a repo option for small models);
select the first setting with at least one success in three development
cases. Main cases have independent, disjoint seeds and are never filtered by
attack success. If both development settings fail, retain the 4% setting,
flag a weak baseline, and report all main outcomes honestly. Clean surrogate
accuracy must exceed 97.5%; otherwise stop with a visible failed status rather
than launching a misleading study. This is a new MNIST experiment, not a
reproduction of the CIFAR-10 numerical results in Witches' Brew.

Two metrics are distinct:

* Primary: intended wrong label on the **chosen target**, averaged over 25
  separate poisoned-data experiments per method (paper-style targeted ASR).
* Secondary: wrong-label fraction on **other test images of the same source
  digit**, excluding the crafted target, averaged over cases (class-wide transfer).

Also record all-10,000-image clean test accuracy, sample SD, 95% Wilson intervals
for target ASR, case-bootstrap intervals for accuracy/group rates, and paired
differences versus standard training. No significance ranking is asserted from
overlapping or unadjusted intervals. Raw predictions for each target case,
class-wide rates, poison tensors, checkpoints, weights and manifests are saved.
Completed model trainings are reused on restart; only an interrupted individual
training restarts. A commit/config mismatch is rejected to prevent mixed results.

Check tomorrow:

```bash
cat artifacts/mnist-lenet-25-study/status.json
cat artifacts/mnist-lenet-25-study/statistics.json
tail -30 artifacts/mnist-lenet-25-study.log
tmux attach -t mnist-defense-25
```

Use Ctrl+B, then D to detach without stopping the job. A completed tmux session
exits; `status.json` will say `complete`. Failures write a traceback to status
and the log. Re-running the same command/commit resumes completed artifacts.
