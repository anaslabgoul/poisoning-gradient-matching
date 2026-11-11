#!/usr/bin/env bash
# A persistent tmux controller submits one scheduler-managed slice at a time.
# It never runs GPU training outside an allocation, nor cancels other jobs.
set -u
cd "${WB_REPO:?Set WB_REPO to the isolated experiment checkout}" || exit 1
export WB_PYTHON="${WB_PYTHON:-$HOME/.venvs/witches-brew/bin/python}"
export WB_RUNNER="${WB_RUNNER:-experiments/run_bilevel_trials.py}"
export WB_ROOT="${WB_ROOT:-$PWD/artifacts/bilevel-resnet18}"
mkdir -p "$WB_ROOT/scheduler"

if [[ "${1:-}" == "worker" ]]; then
    # One GPU worker, even if Slurm inherits a multi-task environment.
    export WB_SLICE_RECEIPT="$WB_ROOT/scheduler/${SLURM_JOB_ID}.python-exit"
    # Some site configurations normalize nonzero srun exits to 1. Preserve the
    # Python status inside the managed step and map an intentional yield to a
    # successful scheduler slice. Infrastructure failures cannot mimic a yield.
    srun --nodes=1 --ntasks=1 --cpus-per-task=4 --exact bash -c '
        "$WB_PYTHON" -u "$WB_RUNNER" \
            --data "$HOME/data" --output "$WB_ROOT" --slice-seconds "${WB_SLICE_SECONDS:-6000}"
        python_result=$?
        printf "%s\n" "$python_result" > "$WB_SLICE_RECEIPT"
        if [[ "$python_result" == "75" ]]; then exit 0; fi
        exit "$python_result"
    '
    exit "$?"
fi

if [[ "${1:-}" == "controller" ]]; then
    export WB_REPO="$PWD"
    exec 9>"$WB_ROOT/controller.lock"
    flock -n 9 || { echo 'Another controller already owns this experiment.'; exit 1; }
    while true; do
        if [[ -n "${WB_HARD_STOP_UTC:-}" ]]; then
            remaining=$("$WB_PYTHON" -c 'import datetime,os,time; print(int(datetime.datetime.fromisoformat(os.environ["WB_HARD_STOP_UTC"]).timestamp()-time.time()))') || exit 1
            if (( remaining <= 180 )); then
                "$WB_PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["WB_ROOT"])/"status.json"; old=json.loads(p.read_text()) if p.exists() else {}; p.write_text(json.dumps(dict(phase="deadline_reached",hard_stop_utc=os.environ["WB_HARD_STOP_UTC"],previous_status=old),indent=2)+"\n")'
                echo 'Morning deadline reached; no further scheduler jobs submitted.'
                exit 0
            fi
        fi
        result=0
        # --wait keeps this controller attached to its own submitted slice.
        # Python exits 75 only after atomically checkpointing its exact position.
        sbatch --wait --parsable --job-name="${WB_JOB_NAME:-wb-bilevel-r18}" --partition=SallesInfo \
            --nodelist=liparis --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 \
            --mem=12G --time=02:00:00 --export=ALL \
            --output="$WB_ROOT/scheduler/slurm-%j.log" \
            "$PWD/experiments/host_bilevel_trials.sh" worker \
            >> "$WB_ROOT/scheduler/jobs.log" 2>> "$WB_ROOT/scheduler/errors.log" || result=$?
        job_id=$(tail -1 "$WB_ROOT/scheduler/jobs.log")
        job_id="${job_id%%;*}"
        python_result=missing
        if [[ -f "$WB_ROOT/scheduler/${job_id}.python-exit" ]]; then
            python_result=$(cat "$WB_ROOT/scheduler/${job_id}.python-exit")
        fi
        phase=$("$WB_PYTHON" -c 'import json,os; p=os.environ["WB_ROOT"]+"/status.json"; print(json.load(open(p)).get("phase","unknown") if os.path.exists(p) else "missing")') || exit 1
        if [[ "$result" == "0" && "$python_result" == "0" && ( "$phase" == "complete" || "$phase" == "complete_with_failures" || "$phase" == "deadline_reached" ) ]]; then
            printf 'Finished: %s\n' "$phase"
            exit 0
        fi
        if [[ "$result" != "0" || "$python_result" != "75" || "$phase" != "checkpointed" ]]; then
            printf 'Stopped after scheduler status %s / Python status %s / experiment phase %s; inspect logs before resuming.\n' "$result" "$python_result" "$phase"
            exit 1
        fi
        printf 'Checkpointed slice; submitting continuation at %s.\n' "$(date -u +%FT%TZ)"
    done
fi

if [[ "${1:-}" != "start" ]]; then
    echo 'Usage: WB_REPO=/absolute/isolated/checkout bash experiments/host_bilevel_trials.sh start'
    exit 2
fi
tmux -L "${WB_TMUX_SOCKET:-wb-bilevel}" new-session -d -s "${WB_TMUX_SESSION:-resnet18}" \
    "cd '$PWD' && bash experiments/host_bilevel_trials.sh controller >> '$WB_ROOT/controller.log' 2>&1"
