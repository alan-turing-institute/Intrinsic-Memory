#!/bin/bash
# One SWE-bench instance image, built natively for aarch64 and migrated to a
# squashfs store on a shared filesystem, so compute nodes read it rather than
# rebuild it.
set -euo pipefail

INSTANCE=${1:?usage: build_swebench_image.sh <instance_id> [task_repo]}
TASK_REPO=${2:-$HOME/swebench-arm64/swe-bench-tasks}
PYTHON=${PYTHON:-python3}
CONTEXTS=${CONTEXTS:-$HOME/swebench-arm64/contexts}
# podman-hpc reads migrated images from here, so a run has to name the same
# directory - tasks/env_configs/swebench_config.yaml does.
export SQUASH_DIR=${SQUASH_DIR:-$HOME/GMemory/swebench-images}

TASK_DIR="$TASK_REPO/tasks/$INSTANCE"
CONTEXT="$CONTEXTS/$INSTANCE"
IMAGE="sweb.eval.arm64.$(printf '%s' "$INSTANCE" | tr '[:upper:]' '[:lower:]'):latest"

[ -d "$TASK_DIR" ] || { echo "no such task: $TASK_DIR" >&2; exit 1; }
mkdir -p "$SQUASH_DIR"

if podman-hpc images --format '{{.Repository}}:{{.Tag}} {{.ReadOnly}}' \
    | grep -qx "localhost/$IMAGE true"; then
    echo "already migrated, nothing to build: $IMAGE"
    exit 0
fi

rm -rf "$CONTEXT"
mkdir -p "$CONTEXT"
# A task Dockerfile may COPY from its own task directory, so that is the context.
cp -R "$TASK_DIR/." "$CONTEXT/"
"$PYTHON" "$(dirname "$0")/swebench_arm64.py" "$TASK_DIR" --out-dir "$CONTEXT"

# -f is resolved inside the build context, so it is named relative to it.
podman-hpc build --layers -f Dockerfile.aarch64 -t "$IMAGE" "$CONTEXT"
podman-hpc migrate "$IMAGE"

# The local image store is erased at the end of a session, so a migrate that
# went somewhere else leaves nothing behind and the next job rebuilds.
podman-hpc images --format '{{.Repository}}:{{.Tag}} {{.ReadOnly}}' \
    | grep -qx "localhost/$IMAGE true" \
    || { echo "migrate did not leave a read-only image; check SQUASH_DIR against \`podman-hpc infohpc\`" >&2; exit 1; }

echo "built and migrated to $SQUASH_DIR: $IMAGE"
