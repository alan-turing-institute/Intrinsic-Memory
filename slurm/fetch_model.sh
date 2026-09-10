#!/bin/bash
# Download a model's weights into the cache the serve jobs read. Run on a login
# node, inside tmux:
#
#   module load brics/tmux
#   tmux new-session -d -s fetch \
#     "MODEL=google/gemma-4-26B-A4B-it slurm/fetch_model.sh > ~/fetch.log 2>&1"
#
# tmux is not a convenience. A login node kills a user's processes when their
# last ssh session to *that node* ends, and ssh to the cluster round-robins
# across login nodes - so `nohup ... &`, `setsid`, and reconnecting to check on
# it all end the same way: no error, no partial download, no process. Under
# tmux the session outlives the shell, and reattaching finds it again.
#
# HF_HOME is under /projects, not $HOME: gemma-4-26B-A4B is 49GB and
# Qwen3.6-35B-A3B is 72GB, and home is quota'd well below either. Filling home
# does not fail the download alone - it leaves $HOME unwritable, so a job that
# only wants to open a log file dies too.
#
# The retry loop resumes from the cache: a download stopped part-way leaves
# .incomplete blobs that look exactly like a finished one, and a later success
# does not clean them up.

set -uo pipefail

MODEL=${MODEL:?set MODEL to the Hub id, e.g. google/gemma-4-26B-A4B-it}
ATTEMPTS=${ATTEMPTS:-40}
export HF_HOME=${HF_HOME:-/projects/u6vh/${USER}/hf}
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "${HF_HOME}"

for attempt in $(seq 1 "${ATTEMPTS}"); do
  echo "=== attempt ${attempt} of ${ATTEMPTS} ==="
  if uv run --quiet --with 'huggingface_hub[cli]' \
      hf download "${MODEL}" --max-workers 2; then
    find "${HF_HOME}" -name '*.incomplete' -delete
    echo "FETCH DONE: ${MODEL} under ${HF_HOME}"
    exit 0
  fi
  echo "attempt ${attempt} stopped short, resuming from the cache"
  sleep 10
done

echo "FETCH FAILED: ${MODEL} still incomplete after ${ATTEMPTS} attempts"
exit 1
