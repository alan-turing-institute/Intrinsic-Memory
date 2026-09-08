# Everything after the server is up. Model-agnostic: the generated preamble
# above sets MODEL_NAME, PORT, DB_DIR, REPO_DIR, TASK, VENV and VLLM_PID.

export OPENAI_API_BASE=http://localhost:${PORT}/v1
export OPENAI_API_KEY="none"

cd "${REPO_DIR}"
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
uv run --no-sync python -c 'import textworld, fast_downward
from alfworld.info import __version__
print("alfworld", __version__, "textworld", textworld.__version__)' \
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
  kill $VLLM_PID 2>/dev/null || true
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
  kill $VLLM_PID 2>/dev/null || true
  exit 1
fi

echo "SMOKE TEST PASSED"

# `wait` on a process the line above killed reports the signal, and under
# `set -e` that ends a passing run non-zero: the log says PASSED and sacct says
# FAILED. Whoever checks the job state rather than reading the log is misled.
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
