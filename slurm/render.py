"""The bodies of the generated slurm scripts.

Four kinds, one serving path: `experiment` (one dataset, every arm),
`crosstask` (every dataset's intrinsic arms, one job), `calibrate` (one dataset,
twenty tasks, to size the others) and `smoke` (the whole path, small).
"""
from config import (
    ACCOUNT,
    CALIBRATE_TIME_LIMIT,
    CPUS_PER_TASK,
    DEFAULT_MAX_TOKENS,
    GPUS,
    MAX_TOKENS_OVERRIDES,
    NODES,
    OMP_NUM_THREADS,
    PORT,
    REPO_DIR,
    SCRATCH_DIR,
    SEEDS,
    SMOKE_TIME_LIMIT,
    TASKS,
    TEMPLATE_DIR,
    TIME_LIMIT,
    VLLM_STARTUP_SLEEP,
    BASELINE_MEMORIES,
    INTRINSIC_ABLATIONS,
    db_dir_for,
    log_dir_for,
)
from models import Model


def intrinsic_memory_for(task: str) -> str:
    return f"intrinsicmemory-{task}"


def intrinsic_arms(task: str) -> list[str]:
    return [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]


def every_arm(task: str) -> list[str]:
    return BASELINE_MEMORIES + intrinsic_arms(task)


def account_directive() -> str:
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def sbatch_header(job_name: str, output_pattern: str, time_limit: str) -> str:
    return f"""#!/bin/bash
{account_directive()}#SBATCH --job-name={job_name}
#SBATCH --nodes={NODES}
#SBATCH --gpus={GPUS}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={output_pattern}
"""


def serve_flags(model: Model) -> str:
    """The `vllm serve` flags this model needs, one per continued line."""
    flags = [
        f'--served-model-name "{model.name}"',
        "--host 0.0.0.0",
        f"--port {PORT}",
        f"--max-model-len {model.max_model_len}",
        f"--max-num-batched-tokens {model.max_num_batched_tokens}",
        f"--max-num-seqs {model.max_num_seqs}",
        f"--tensor-parallel-size {GPUS}",
    ]
    if model.yaml_config:
        flags.insert(1, f'--config "{model.yaml_config}"')
    if model.reasoning_parser:
        flags.append(f"--reasoning-parser {model.reasoning_parser}")
    flags.extend(model.extra_serve_flags)

    return " \\\n    ".join(flags)


def serve_block(model: Model) -> str:
    """Start vLLM and wait for it, or say why it never came up.

    The wait has to end when the server dies as well as when it answers: a vLLM
    that fails to start never serves /health, and the loop alone would spend the
    whole allocation waiting for it.
    """
    environment = "".join(
        f'export {name}="{value}"\n' for name, value in
        ((("HF_HOME", model.hf_home),) if model.hf_home else ())
        + ((("TIKTOKEN_ENCODINGS_BASE", model.tiktoken_encodings),)
           if model.tiktoken_encodings else ())
        + model.extra_env
    )

    return f"""{environment}
cd {model.vllm_dir}
source .venv/bin/activate
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'

srun \\
    --nodes=$SLURM_NNODES \\
    --gpus=$SLURM_GPUS \\
    --cpus-per-task {CPUS_PER_TASK} \\
    --ntasks-per-node 1 \\
    vllm serve "{model.served}" \\
    {serve_flags(model)} &

VLLM_PID=$!

until curl -s http://localhost:{PORT}/health > /dev/null 2>&1; do
  if ! kill -0 ${{VLLM_PID}} 2>/dev/null; then
    echo "FAILED: vLLM exited before it answered /health"
    wait ${{VLLM_PID}} || true
    exit 1
  fi
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"
curl -s http://localhost:{PORT}/v1/models

deactivate"""


TMPDIR_PROBE = """# The per-user node-local scratch that TMPDIR names is created by the job prolog.
# A node where that failed leaves TMPDIR naming a directory nothing can write, and
# vLLM reports it as a PermissionError from a worker rather than as a bad node.
echo -n "TMPDIR: "
if mkdir -p "${TMPDIR:-/tmp}" 2>/dev/null && touch "${TMPDIR:-/tmp}/.probe" 2>/dev/null; then
  rm -f "${TMPDIR:-/tmp}/.probe"
  echo "${TMPDIR:-/tmp}"
else
  export TMPDIR="${SCRATCH}/fallback-${SLURM_JOB_ID:-local}"
  mkdir -p "${TMPDIR}"
  echo "unwritable on $(hostname), falling back to ${TMPDIR}"
fi

# vLLM asks for every GPU by tensor_parallel_size, and a step that cannot see
# them all fails inside torch as `device >= 0 && device < num_gpus INTERNAL
# ASSERT FAILED`, which names neither the count nor the step. Count them first.
echo -n "gpus visible to a step: "
srun --nodes=1 --gpus=${SLURM_GPUS} --ntasks-per-node 1 \\
  bash -c 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null || echo "could not launch a step"
"""


def run_command(task: str, memories: list[str], cross_task: bool, model: Model,
                background: bool = False, seeds: list[int] = SEEDS, scope: str = "") -> str:
    """One `tasks/run.py` invocation, for one dataset and its arms.

    A backgrounded one records its own pid: a bare `wait` would also wait on the
    vLLM server started the same way, which only ever exits when killed.

    `--resume` makes a job that is submitted again finish what the last one ran
    out of time for. It skips whichever experiments already have a row in
    ${DB_DIR}, so pointing a job at a directory whose results you meant to
    replace will skip them rather than redo them - use a new sweep for that.
    """
    flag = "\n\t--intrinsic_cross_task \\" if cross_task else ""
    trailing = " &\nRUN_PIDS+=($!)" if background else ""

    return f"""uv run --no-sync tasks/run.py \\
\t--task {task} \\
\t--mas_type autogen \\
\t--mas_memory {" ".join(memories)} \\
\t--seed {" ".join(str(seed) for seed in seeds)} \\{flag}{scope}
\t--db_dir ${{DB_DIR}} \\
\t--model ${{MODEL_NAME}} \\
\t--resume \\
\t--max_tokens {MAX_TOKENS_OVERRIDES.get(task, DEFAULT_MAX_TOKENS)}{trailing}"""


def preamble(model: Model, job_name: str, output_pattern: str, script_name: str, *,
             db_dir: str, time_limit: str = TIME_LIMIT, probes: bool = False,
             scratch_dir: str = SCRATCH_DIR) -> str:
    """Everything before the run: the allocation, the server, the environment."""
    return f"""{sbatch_header(job_name, output_pattern, time_limit)}
echo "SERVING {model.name} ON $HOSTNAME"

module reset
module load brics/nccl
module list

# Every job of one experiment set must point at the same directory: they append to
# one overall_results.csv under a lock on the file. A new sweep needs a new one:
#   uv run slurm/generate_slurm.py --sweep <date>
# or override this one at submit time:
#   DB_DIR=<dir> sbatch slurm/generated/{model.slug}/{script_name}
DB_DIR=${{DB_DIR:-{db_dir}}}
REPO_DIR={REPO_DIR}
SCRATCH={scratch_dir}
MODEL_NAME="{model.name}"
PORT={PORT}
{TMPDIR_PROBE if probes else ""}
{serve_block(model)}

# experiment setup
export MODEL_NAME
export OPENAI_API_BASE=http://localhost:{PORT}/v1
export OPENAI_API_KEY="none"
export OMP_NUM_THREADS={OMP_NUM_THREADS}

cd {REPO_DIR}
source .venv/bin/activate

sleep {VLLM_STARTUP_SLEEP}

# The experiment processes only, not vLLM: vLLM keeps the node-local TMPDIR it
# started with, which is faster and big enough for one process.
export TMPDIR="${{SCRATCH}}/{job_name}-${{SLURM_JOB_ID}}"
mkdir -p "${{TMPDIR}}"
trap 'rm -rf "${{TMPDIR}}"' EXIT TERM INT
echo "TMPDIR -> ${{TMPDIR}} ($(df -h "${{TMPDIR}}" | tail -1 | awk '{{print $4}}') free)"

echo "results -> ${{DB_DIR}}"
"""


CLEANUP = """
# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
"""


def render_experiment(model: Model, task: str, sweep: str, seeds: list[int] = SEEDS) -> str:
    return (
        preamble(
            model, f"vllm-{task}", f"{log_dir_for(sweep)}/{task}-%x.%j.%t.out",
            f"{task}_experiment.sh", db_dir=db_dir_for(sweep),
        )
        + "\n"
        + run_command(task, every_arm(task), cross_task=False, model=model, seeds=seeds)
        + CLEANUP
    )


def render_crosstask(model: Model, sweep: str, tasks: list[str] = TASKS,
                     seeds: list[int] = SEEDS) -> str:
    """Every dataset's cross-task arms, in one job against one server.

    The datasets get a `run.py` each rather than one sweep over all of them:
    the sweep is a Cartesian product, so a single call would pair every dataset
    with every other dataset's hand-written template. They run concurrently and
    append to the same files under the lock that already makes two submitted
    jobs safe.
    """
    runs = "\n\n".join(
        run_command(task, intrinsic_arms(task), cross_task=True, model=model,
                    background=True, seeds=seeds)
        for task in tasks
    )

    return (
        preamble(
            model, "vllm-crosstask", f"{log_dir_for(sweep)}/crosstask-%x.%j.%t.out",
            "crosstask.sh", db_dir=db_dir_for(sweep),
        )
        + """
# The cross-task arm: an intrinsic memory is kept across the tasks of the dataset
# instead of starting each task from an empty one. Only the intrinsicmemory-* modules
# read the flag, and the same --db_dir as the baseline is deliberate - the two arms are
# told apart by the intrinsic_cross_task column, not by the file they are in.

RUN_PIDS=()

"""
        + runs
        + '\n\nwait "${RUN_PIDS[@]}"\n'
        + CLEANUP
    )


def render_calibration(model: Model, task: str, sweep: str, *, seeds: list[int],
                       db_dir: str, scope: str,
                       time_limit: str = CALIBRATE_TIME_LIMIT) -> str:
    """One dataset, every arm, one seed, a few tasks - to size the real job.

    The result rows carry the token spend, so tokens / elapsed = throughput, and
    episodes x tokens-per-task / throughput = the wall clock a real job needs.
    """
    return (
        preamble(
            model, f"vllm-{task}-calibrate",
            f"{log_dir_for(sweep)}/{task}-calibrate-%x.%j.%t.out",
            f"{task}_calibrate.sh", db_dir=db_dir, time_limit=time_limit,
        )
        + "\n"
        + run_command(task, every_arm(task), cross_task=False, model=model,
                      seeds=seeds, scope=scope)
        + CALIBRATION_SUMMARY
        + CLEANUP
    )


CALIBRATION_SUMMARY = """

echo "==== calibration ===="
column -s, -t < ${DB_DIR}/overall_results.csv

python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tokens = sum(int(r["completion_tokens"]) + int(r["prompt_tokens"]) for r in rows)
scored = sum(int(r["tasks_scored"]) for r in rows)
print(f"{len(rows)} arms, {scored} tasks scored, {tokens:,} tokens")
print(f"{tokens/max(scored, 1):,.0f} tokens per task")
' ${DB_DIR}/overall_results.csv

cat ${DB_DIR}/*/*/*/*/failed_tasks.csv 2>/dev/null
"""


def render_smoke(model: Model, sweep: str, task: str = "fever") -> str:
    """The whole path, small: serve, run two tasks through two memory modules, check.

    Worth a submission before any 24-hour job, and after any change to the
    cluster, the model or the environment. It is also how a model is brought up
    for the first time - it is the only script that asserts the endpoint answers
    in the shape the experiment needs.
    """
    header = sbatch_header(
        "vllm-smoke", f"{log_dir_for(sweep)}/smoke-%x.%j.%t.out", SMOKE_TIME_LIMIT,
    )
    body = (TEMPLATE_DIR / "smoke_body.sh").read_text()

    return f"""{header}
# Defaults to {task}. Any other dataset with
#   TASK=alfworld sbatch slurm/generated/{model.slug}/smoke_test.sh
#
# VENV picks the environment the run uses, so a dataset whose simulator is not in
# the shared one can be checked against a venv of its own without disturbing a
# queued job that shares it:
#   VENV=~/alfworld-test-venv TASK=alfworld sbatch ...

set -euo pipefail

TASK=${{TASK:-{task}}}
VENV=${{VENV:-.venv}}
MODEL_NAME="{model.name}"
PORT={PORT}
REPO_DIR={REPO_DIR}
SCRATCH={SCRATCH_DIR}
DB_DIR=${{DB_DIR:-./.db/smoke-${{TASK}}-${{SLURM_JOB_ID:-local}}}}

echo "SERVING {model.name} ON $HOSTNAME"

module reset
module load brics/nccl
module list

{TMPDIR_PROBE}
{serve_block(model)}

{body}"""
