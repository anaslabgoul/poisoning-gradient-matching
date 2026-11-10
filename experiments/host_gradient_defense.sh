#!/usr/bin/env bash
#SBATCH --job-name=wb-gradient-defense
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=liparis
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/gradient-defense-25/slurm-%j.log

set -u
cd "${WB_REPO:?Set WB_REPO}" || exit 1
export WB_ROOT="$PWD/artifacts/gradient-defense-25"
export WB_PYTHON="$HOME/.venvs/witches-brew/bin/python"
export WB_SOCKET="wb-gradient-${SLURM_JOB_ID}"
export WB_EXIT_FILE="$WB_ROOT/scheduler/${SLURM_JOB_ID}.exit"

if [[ "${1:-}" == "body" ]]; then
    result=0
    if [[ "${WB_PROBE:-0}" == "1" ]]; then
        srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=4 --exact /bin/sleep 60 || result=$?
    else
        # srun registers the Python worker as a managed Slurm step even though
        # its tmux controller is detached; cancelling the job kills the worker.
        srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=4 --exact \
            "$WB_PYTHON" -u experiments/compare_saved_gradient.py --data "$HOME/data" \
            > "$WB_ROOT/run.log" 2>&1 || result=$?
    fi
    printf '%s\n' "$result" > "$WB_EXIT_FILE"
    tmux -L "$WB_SOCKET" wait-for -S defense_done
    exit "$result"
fi

mkdir -p "$WB_ROOT/scheduler"
cleanup() { tmux -L "$WB_SOCKET" kill-server >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT
tmux -L "$WB_SOCKET" new-session -d -s gradient-defense "bash '$PWD/experiments/host_gradient_defense.sh' body" || exit 1
tmux -L "$WB_SOCKET" wait-for defense_done
if [[ -f "$WB_EXIT_FILE" ]]; then
    exit "$(cat "$WB_EXIT_FILE")"
fi
exit 99
