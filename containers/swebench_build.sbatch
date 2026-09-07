#!/bin/bash
# Build one SWE-bench instance image as a batch job.
#
#   sbatch containers/swebench_build.sbatch <instance_id>
#
# A login-node build does not survive logout: linger is denied for user
# accounts, so logind kills the build, and tmux with it. Compute nodes have
# outbound network and podman-hpc, and `migrate` writes to $SCRATCH, so a job
# can do the whole thing.
#SBATCH --job-name=sweb-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
set -euo pipefail

INSTANCE=${1:?usage: sbatch swebench_build.sbatch <instance_id>}
WORK=${WORK:-$HOME/swebench-arm64}

module reset

cd "$WORK"
PYTHON=${PYTHON:-$WORK/.venv/bin/python} ./build_swebench_image.sh "$INSTANCE"
