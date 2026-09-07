#!/bin/bash
# One SWE-bench instance image, built natively for aarch64 and migrated to
# $SCRATCH so compute nodes can see it.
set -euo pipefail

INSTANCE=${1:?usage: build_swebench_image.sh <instance_id> [task_repo]}
TASK_REPO=${2:-$HOME/swebench-arm64/swe-bench-tasks}
PYTHON=${PYTHON:-python3}
CONTEXTS=${CONTEXTS:-$HOME/swebench-arm64/contexts}

TASK_DIR="$TASK_REPO/tasks/$INSTANCE"
CONTEXT="$CONTEXTS/$INSTANCE"
IMAGE="sweb.eval.arm64.$(printf '%s' "$INSTANCE" | tr '[:upper:]' '[:lower:]'):latest"

[ -d "$TASK_DIR" ] || { echo "no such task: $TASK_DIR" >&2; exit 1; }

rm -rf "$CONTEXT"
mkdir -p "$CONTEXT"
# A task Dockerfile may COPY from its own task directory, so that is the context.
cp -R "$TASK_DIR/." "$CONTEXT/"
"$PYTHON" "$(dirname "$0")/swebench_arm64.py" "$TASK_DIR" --out-dir "$CONTEXT"

# -f is resolved inside the build context, so it is named relative to it.
podman-hpc build --layers -f Dockerfile.aarch64 -t "$IMAGE" "$CONTEXT"
podman-hpc migrate "$IMAGE"

echo "built and migrated: $IMAGE"
