#!/usr/bin/env bash
#SBATCH --job-name=wb-lambda-sweep
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=liparis
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%1
#SBATCH --output=artifacts/gradient-lambda-sweep/slurm-%A-%a.log

set -u
cd "${WB_REPO:?Set WB_REPO}" || exit 1
export WB_PYTHON="$HOME/.venvs/witches-brew/bin/python"
export WB_SOCKET="wb-lambda-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
# The frozen plan is the only source of lambda and output-directory values.
setting=$("$WB_PYTHON" -c 'import json,os; s=json.load(open("experiments/protocols/gradient-lambda-sweep.json"))["settings"][int(os.environ["SLURM_ARRAY_TASK_ID"])]; print(s["lambda"],s["directory"])') || exit 1
read -r WB_LAMBDA_WEIGHT WB_DIRECTORY <<< "$setting"
export WB_LAMBDA_WEIGHT WB_DIRECTORY
export WB_ROOT="$PWD/artifacts/gradient-lambda-sweep/$WB_DIRECTORY"
export WB_EXIT_FILE="$WB_ROOT/scheduler/${SLURM_JOB_ID}.exit"

if [[ "${1:-}" == "body" ]]; then
    result=0
    srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=4 --exact \
        "$WB_PYTHON" -u experiments/compare_saved_gradient.py --data "$HOME/data" \
        --lambda-weight "$WB_LAMBDA_WEIGHT" --output "$WB_ROOT" \
        > "$WB_ROOT/run.log" 2>&1 || result=$?
    "$WB_PYTHON" experiments/summarize_lambda_sweep.py > "$WB_ROOT/summary-update.log" 2>&1 || result=$?
    printf '%s\n' "$result" > "$WB_EXIT_FILE"
    tmux -L "$WB_SOCKET" wait-for -S sweep_done
    exit "$result"
fi

mkdir -p "$WB_ROOT/scheduler"
cleanup() { tmux -L "$WB_SOCKET" kill-server >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT
tmux -L "$WB_SOCKET" new-session -d -s lambda-sweep "bash '$PWD/experiments/host_gradient_lambda_sweep.sh' body" || exit 1
tmux -L "$WB_SOCKET" wait-for sweep_done
if [[ -f "$WB_EXIT_FILE" ]]; then
    exit "$(cat "$WB_EXIT_FILE")"
fi
exit 99
