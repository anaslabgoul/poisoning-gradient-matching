# MNIST / LeNet overnight statistics

## Completed run outcome

The completed 100-victim run is summarized in
[`results/mnist-lenet-25-summary.json`](results/mnist-lenet-25-summary.json).
All four methods had 0/25 intended-target successes, including standard training.
Consequently this configuration did **not** provide a useful defense-robustness
comparison. The low observed success rates must not be attributed to defenses.
The primary 95% Wilson interval is 0--13.32% for each method; degenerate paired
ASR bootstrap intervals do not establish equivalence. Full raw measurements
remain in `artifacts/mnist-lenet-25-study/statistics.json` on liparis and locally.

### Baseline diagnostics and budget tuning

`results/mnist-baseline-diagnosis.json` records an audit of all 25 poison tensors
and three additional one-case standard-training controls. All 2,400 selected
training rows changed in the checked case; saved IDs/labels/bounds matched in
all 25 cases. Clean target 2 stayed 2; bounded poison stayed 2 even when the
victim used the source initialization. A deliberately invalid positive control
(inserting the target with wrong label, outside the noise bound) changed it to
7 with probability 0.998515. This verifies the basic training/scoring path,
not attack effectiveness under the required constraints. The exact reason for
the bounded attack's failure is not yet established.

`experiments/tune_standard_budget.py` runs only standard-training development
cases across explicitly requested budgets/objectives, with matched targets and
seeds and clean controls. It records actual poison counts because the single-
class threat model cannot use more poisons than that class contains (about 10%
on MNIST). Example maximum-budget diagnostic:

```bash
python experiments/tune_standard_budget.py --output artifacts/mnist-standard-budget-max --budgets .10 --losses similarity SE --source artifacts/mnist-lenet-25-study/preparation/models/surrogate.pt
```

The [paper's Table 2](https://arxiv.org/html/2009.02276v2#S5.T2) reports 86.25%
targeted success on CIFAR-10 with its 6-layer ConvNet, 1% budget and epsilon
32/255. Its ConvNet has widths 64,128,128,256,256 and a linear head (`ConvNet64`
in this repo), and uses 40 epochs. These are reference settings, not a promise
that changing MNIST's poison fraction alone will reproduce that success.
Keep development selection separate from a new independent final evaluation.

### Completed local failure investigation (10 November 2026)

`results/local-attack-investigation.json` preserves the local RTX 3050 diagnosis.
The detailed interventions use the preselected failed case 3100000000 (2 to 7),
not a newly selected successful target. All 2,400 saved modifications were
applied to the correct IDs, with unchanged label 7 and pixel bound 32/255.
The source gradient direction changes as intended and a small poison-only
gradient update lowers its target loss. The matched poison-trained victim
nevertheless fits 99.92% of those modified training images correctly and keeps
the target classified as 2. The clean-trained victim recognizes 62.79% of those
same modified images. Source gradient matching is therefore present, but it
does not produce the desired final target error in this setup.

Six local 20-epoch Tanh controls compared clean/poisoned data with the original
training setup, an actual trained-source warm start, and translations disabled.
Every evaluated epoch retained target label 2. Replacing Tanh by ReLU, keeping
61,706 parameters and all attack settings, then freshly optimizing the poison,
also failed in this case. Those individual interventions did not rescue it;
they do not establish a unique cause for all 25 original failures or inherent
robustness. The random approximation of the defense regularizer is irrelevant
to the standard condition, which contains no such penalty.

Local commands (saved input checkpoints/data are required):

```powershell
.\.venv\Scripts\python.exe experiments\investigate_local_attack.py
.\.venv\Scripts\python.exe experiments\test_mnist_activation.py --output artifacts/local-activation-test-new
```

Windows uses its supported multiprocessing sharing strategy; Linux retains
the configured file-descriptor strategy. A project-local CUDA PyTorch 2.8.0 /
torchvision 0.23.0 environment was used. Ten relevant CPU checks and a full
local 250-iteration, eight-restart GPU brewing run completed successfully.
No new full defense study was launched.

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
select the first setting with at least two successes in three development
cases. Main cases have independent, disjoint seeds and are never filtered by
attack success. If all three development settings fail this operational quality
gate, STOP before any main/defended runs. The completed historical run did not
enforce this gate and incorrectly proceeded after all nine pilot failures.
Those results are preserved, not relabeled as a successful benchmark. A 2/3
development pass is a screening rule, not an estimate of final attack success.
Clean surrogate
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
