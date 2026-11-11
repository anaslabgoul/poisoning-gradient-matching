#!/usr/bin/env bash
# A persistent tmux controller submits one scheduler-managed slice at a time.
# It never runs GPU training outside an allocation, nor cancels other jobs.
set -u
cd "${WB_REPO:?Set WB_REPO to the isolated experiment checkout}" || exit 1
export WB_PYTHON="${WB_PYTHON:-$HOME/.venvs/witches-brew/bin/python}"
export WB_ROOT="$PWD/artifacts/bilevel-resnet18"
mkdir -p "$WB_ROOT/scheduler"

if [[ "${1:-}" == "worker" ]]; then
    # One GPU worker, even if Slurm inherits a multi-task environment.
    srun --nodes=1 --ntasks=1 --cpus-per-task=4 --exact \
        "$WB_PYTHON" -u experiments/run_bilevel_trials.py \
        --data "$HOME/data" --output "$WB_ROOT" --slice-seconds 6000
    exit "$?"
fi

if [[ "${1:-}" == "controller" ]]; then
    export WB_REPO="$PWD"
    exec 9>"$WB_ROOT/controller.lock"
    flock -n 9 || { echo 'Another controller already owns this experiment.'; exit 1; }
    while true; do
        result=0
        # --wait keeps this controller attached to its own submitted slice.
        # Python exits 75 only after atomically checkpointing its exact position.
        sbatch --wait --parsable --job-name=wb-bilevel-r18 --partition=SallesInfo \
            --nodelist=liparis --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 \
            --mem=12G --time=02:00:00 --export=ALL \
            --output="$WB_ROOT/scheduler/slurm-%j.log" \
            "$PWD/experiments/host_bilevel_trials.sh" worker \
            >> "$WB_ROOT/scheduler/jobs.log" 2>&1 || result=$?
        phase=$("$WB_PYTHON" -c 'import json,os; p=os.environ["WB_ROOT"]+"/status.json"; print(json.load(open(p)).get("phase","unknown") if os.path.exists(p) else "missing")') || exit 1
        if [[ "$result" == "0" && ( "$phase" == "complete" || "$phase" == "complete_with_failures" ) ]]; then
            printf 'Finished: %s\n' "$phase"
            exit 0
        fi
        if [[ "$result" != "75" || "$phase" != "checkpointed" ]]; then
            printf 'Stopped after scheduler status %s / experiment phase %s; inspect logs before resuming.\n' "$result" "$phase"
            exit 1
        fi
        printf 'Checkpointed slice; submitting continuation at %s.\n' "$(date -u +%FT%TZ)"
    done
fi

if [[ "${1:-}" != "start" ]]; then
    echo 'Usage: WB_REPO=/absolute/isolated/checkout bash experiments/host_bilevel_trials.sh start'
    exit 2
fi
tmux -L wb-bilevel new-session -d -s resnet18 \
    "cd '$PWD' && bash experiments/host_bilevel_trials.sh controller >> '$WB_ROOT/controller.log' 2>&1"
