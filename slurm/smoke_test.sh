#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=out/smoke-%x.%j.%t.out

# The whole path, small: serve, run two tasks of one dataset through two memory
# modules, and check what came out. Run this before submitting a 24-hour job, and
# after any change to the cluster, the model or the environment.
#
# Defaults to fever. Any other dataset with
#   TASK=alfworld sbatch slurm/smoke_test.sh
#
# VENV picks the environment the run uses, so a dataset whose simulator is not in
# the shared one can be checked against a venv of its own without disturbing a
# queued job that shares it:
#   VENV=~/alfworld-test-venv TASK=alfworld sbatch slurm/smoke_test.sh

set -euo pipefail

TASK=${TASK:-fever}
VENV=${VENV:-.venv}

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

MODEL_NAME="openai/gpt-oss-120b"
YAML_CONFIG="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME=/projects/public/brics/hf
MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/
DB_DIR=./.db/smoke-${TASK}-${SLURM_JOB_ID:-local}

export TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"

cd ~/vllm_test
source .venv/bin/activate

srun \
    --nodes=$SLURM_NNODES \
    --gpus=$SLURM_GPUS \
    --cpus-per-task 16 \
    --ntasks-per-node 1 \
    vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --config $YAML_CONFIG \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 512 \
    --tensor_parallel_size=4 &

VLLM_PID=$!

# The wait has to end when the server dies as well as when it answers: a vLLM
# that fails to start never serves /health, and the loop alone would spend the
# whole allocation waiting for it.
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "SMOKE TEST FAILED: vLLM exited before it answered /health"
    wait ${VLLM_PID} || true
    exit 1
  fi
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"
deactivate

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="none"

cd ~/GMemory
# `uv run` ignores an activated venv that is not the project's own and uses .venv
# regardless, with a warning - so the choice has to be made through uv's variable.
export UV_PROJECT_ENVIRONMENT="${VENV}"
source ${VENV}/bin/activate

# The result files are appended to under an flock, which a filesystem has to be
# mounted for. Lustre supports it with the flock mount option; without it,
# separately submitted jobs can interleave their writes.
mkdir -p ${DB_DIR}
if flock -n ${DB_DIR}/.flock-probe true 2>/dev/null; then
  echo "flock: supported on $(df -T ${DB_DIR} 2>/dev/null | tail -1 || df ${DB_DIR} | tail -1)"
else
  echo "flock: REFUSED - jobs writing to one results file may interleave"
fi

# FEVER and HotpotQA reach live Wikipedia through Search, so those tasks need
# outbound network from the compute node. Without it every claim fails and the run
# still writes rows - which is why the assertion below is on tasks_scored, not just
# on the row count.
echo -n "wikipedia reachable: "
curl -s -o /dev/null -w '%{http_code}\n' --max-time 20 \
  https://en.wikipedia.org/api/rest_v1/page/summary/Water || echo "unreachable"

# ScienceWorld runs a JVM through py4j, so the sciworld jobs need java on PATH.
echo -n "java: "; java -version 2>&1 | head -1 || echo "absent - sciworld will not start"

# ALFWorld plays a .tw-pddl game file per task, and neither its simulator nor the
# games are installed by `uv sync`; see the ALFWorld section of data/data.md.
echo -n "alfworld: "
uv run --no-sync python -c 'import alfworld, textworld; print("simulator", alfworld.__version__)' \
  2>&1 | tail -1 || echo "absent - alfworld will not start"
echo -n "alfworld games: "
find data/alfworld -name 'game.tw-pddl' 2>/dev/null | wc -l

uv run --no-sync tasks/run.py \
	--task ${TASK} \
	--mas_type autogen \
	--mas_memory empty intrinsicmemory-notemplate \
	--seed 11 \
	--max_tasks 2 \
	--max_trials 3 \
	--db_dir ${DB_DIR} \
	--model ${MODEL_NAME}

echo "==== what the run wrote ===="
find ${DB_DIR} -name '*.csv' | sort
echo "==== overall_results.csv ===="
cat ${DB_DIR}/overall_results.csv

# 2 memory modules, so 2 rows plus a header. Fewer means an experiment failed.
rows=$(($(wc -l < ${DB_DIR}/overall_results.csv) - 1))
if [ "$rows" -ne 2 ]; then
  echo "SMOKE TEST FAILED: expected 2 result rows, got ${rows}"
  cat ${DB_DIR}/failed_experiments.csv 2>/dev/null
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# An experiment whose every task failed still writes a row, with tasks_scored 0 - the
# shape of a task that cannot reach what it needs, rather than of a broken sweep.
unscored=$(python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(sum(1 for r in rows if int(r["tasks_scored"]) == 0))
' ${DB_DIR}/overall_results.csv)
if [ "$unscored" -ne 0 ]; then
  echo "SMOKE TEST FAILED: ${unscored} experiments scored no tasks at all"
  find ${DB_DIR} -name 'failed_tasks.csv' -exec cat {} +
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

echo "SMOKE TEST PASSED"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
