#!/bin/bash
# Build one SWE-bench instance image as a batch job.
#
#   sbatch containers/swebench_build.sbatch <instance_id>
#   sbatch --array=0-19 containers/swebench_build.sbatch <file of instance ids>
#
# A login-node build does not survive logout: linger is denied for user
# accounts, so logind kills the build, and tmux with it. Compute nodes have
# outbound network and podman-hpc, and `migrate` writes to a shared filesystem,
# so a job can do the whole thing.
#SBATCH --job-name=sweb-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
set -euo pipefail

TARGET=${1:?usage: sbatch swebench_build.sbatch <instance_id | file of instance ids>}
# An array job takes its instance from the line the array index names, so one
# submission builds a whole set.
if [ -f "$TARGET" ]; then
    INSTANCE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$TARGET")
    [ -n "$INSTANCE" ] || { echo "no instance on line $((${SLURM_ARRAY_TASK_ID:-0} + 1)) of $TARGET" >&2; exit 1; }
else
    INSTANCE=$TARGET
fi
WORK=${WORK:-$HOME/swebench-arm64}
# The images outlive this job here, and the experiment reads them from the same
# directory rather than building them again on the node it runs on.
export SQUASH_DIR=${SQUASH_DIR:-$HOME/GMemory/swebench-images}

module reset

cd "$WORK"
PYTHON=${PYTHON:-$WORK/.venv/bin/python} ./build_swebench_image.sh "$INSTANCE"
