#!/usr/bin/env bash
#SBATCH --job-name=wb-fixed-grid
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=liparis
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/fixed-targets-10x10/slurm-%A_%a.log

set -u
cd "${WB_REPO:?Set WB_REPO to the remote repository}" || exit 1
export WB_ROOT="$PWD/artifacts/fixed-targets-10x10"
export WB_PYTHON="$HOME/.venvs/witches-brew/bin/python"

if [[ "${1:-}" == "body" ]]; then
    result=0
    if [[ "$WB_MODE" == "prepare" ]]; then
        "$WB_PYTHON" -u experiments/fixed_target_grid.py --mode prepare --data "$HOME/data" --output "$WB_ROOT" > "$WB_ROOT/prepare.log" 2>&1 || result=$?
    else
        "$WB_PYTHON" -u experiments/fixed_target_grid.py --mode target --target-index "$SLURM_ARRAY_TASK_ID" --data "$HOME/data" --output "$WB_ROOT" > "$WB_ROOT/target-$SLURM_ARRAY_TASK_ID.log" 2>&1 || result=$?
    fi
    printf '%s\n' "$result" > "$WB_EXIT_FILE"
    tmux -L "$WB_SOCKET" wait-for -S grid_done
    exit "$result"
fi

export WB_SOCKET="wb-grid-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-prepare}"
mkdir -p "$WB_ROOT/scheduler"
export WB_EXIT_FILE="$WB_ROOT/scheduler/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-prepare}.exit"
tmux -L "$WB_SOCKET" new-session -d -s fixed-grid "bash '$PWD/experiments/host_fixed_grid.sh' body" || exit 1
tmux -L "$WB_SOCKET" wait-for grid_done
if [[ -f "$WB_EXIT_FILE" ]]; then
    exit "$(cat "$WB_EXIT_FILE")"
fi
exit 99
