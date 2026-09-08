#!/usr/bin/env python3
"""Generate the slurm/*_experiment.sh sweep scripts and slurm/crosstask.sh.

The generated scripts are not committed; this generator is. Edit the constants
below and rerun to produce a new sweep:

    uv run slurm/generate_slurm.py                    # a sweep dated today
    uv run slurm/generate_slurm.py --sweep 2026-09-08  # rejoin an existing one

The calibration runs that size these jobs are slurm/generate_calibration.py,
which builds its scripts from the cluster configuration and the job pieces here.

Set SLURM_ACCOUNT to name an account in the generated scripts; without it they
submit under the user's default.
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

SLURM_DIR = Path(__file__).parent

# --- experiment matrix ---------------------------------------------------
# One experiment script per task runs its TASKS x BASELINE_MEMORIES x SEEDS
# combinations plus the intrinsic ablations: 10 arms x 10 seeds. One crosstask
# script covers every task's intrinsic arms again: 7 x 3 arms x 10 seeds.

TASKS = ["alfworld", "babyai", "fever", "hotpotqa", "jericho", "pddl", "sciworld"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# Non-intrinsic baselines, run in the experiment script only.
BASELINE_MEMORIES = [
    "empty", "chatdev", "voyager", "memorybank", "generative", "metagpt", "g-memory",
]
# Intrinsic ablations that don't vary by task; the per-task intrinsic memory
# (intrinsicmemory-<task>) is added alongside these in both scripts.
INTRINSIC_ABLATIONS = ["intrinsicmemory-notemplate", "intrinsicmemory-llm-structured-template"]

# vLLM's request queue depth. A job's requests in flight are its experiment
# count, which is at most 210 - the crosstask job, seven datasets x 3 arms x 10
# seeds - and the KV cache holds 3,730,336 tokens, 227 of them at MAX_MODEL_LEN.
# Past that the depth is a number the server cannot honour.
MAX_NUM_SEQS = 256

# The budget a call starts at, per task; tasks not listed use DEFAULT_MAX_TOKENS.
# A starved reasoning model is retried with a doubled budget, so too small a
# start is paid for in whole wasted calls rather than in a truncated answer.
MAX_TOKENS_OVERRIDES = {"babyai": 4096, "pddl": 4096}
DEFAULT_MAX_TOKENS = 2048

# --- cluster / model configuration ---------------------------------------

REPO_DIR = "~/GMemory"
VLLM_VENV_DIR = "~/vllm_test"
YAML_CONFIG = "/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME = "/projects/public/brics/hf"
MODEL_SNAPSHOT = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_PATH = f"{HF_HOME}/hub/models--openai--gpt-oss-120b/snapshots/{MODEL_SNAPSHOT}/"
MODEL_NAME = "openai/gpt-oss-120b"
TIKTOKEN_ENCODINGS_BASE = "/projects/public/brics/distributed_vllm/etc/encodings"
# One user's project allocation, and the root of everything a job writes.
# Home is a 101 G quota; one sweep's results are 17 G, its logs 15 G, and the
# scratch ALFWorld churns through is larger than the quota on its own.
PROJECT_DIR = "/projects/u6vh/syuen.u6vh"

# The sweep a script belongs to, dating its results and logs so a new one lands
# beside the last rather than on top of it: a run refuses to append to a results
# file whose header is not its schema.
#
# Today by default, and `--sweep` to name an existing one. Pass it whenever
# regenerating scripts for a sweep already under way - every job of a sweep has
# to name the same results directory for `--resume` to see what the last one
# finished, and a sweep outlives the day its scripts were generated on.
DEFAULT_SWEEP = date.today().isoformat()
# Where the experiment processes put their temporary files. Not the node-local
# scratch TMPDIR names by default: ALFWorld's PDDL engine copies a 28.8 MB
# libdownward.so into it on every environment load, and 100 experiments doing
# that at once exhausted it - 10,358 tasks of one sweep died on ENOSPC.
SCRATCH_DIR = f"{PROJECT_DIR}/tmp"

def db_dir_for(sweep: str) -> str:
    return f"{PROJECT_DIR}/results/sweep-{sweep}"


def log_dir_for(sweep: str) -> str:
    return f"{PROJECT_DIR}/logs/{sweep}"


NODES = 1
GPUS = 4
TENSOR_PARALLEL_SIZE = 4
CPUS_PER_TASK = 16
TIME_LIMIT = "24:00:00"
PORT = 8000
VLLM_STARTUP_SLEEP = 100

# Named at generation time rather than committed: it is one user's allocation.
ACCOUNT = os.environ.get("SLURM_ACCOUNT")

# These three override the shared GPT-OSS_Hopper.yaml, which sets them to 8192,
# 10240 and off.
#
# MAX_NUM_BATCHED_TOKENS is a per-step total across the whole batch, not a
# per-request cap, so being the larger of the two is what lets a step prefill
# several requests at once instead of one at a time. MAX_MODEL_LEN is per
# request, and has to clear the largest prompt plus --max_tokens_ceiling.
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 16384
ENABLE_PREFIX_CACHING = True

# One worker process per experiment, each loading an embedding model on the CPU.
# Unset, every one of them sizes its thread pool to the whole node.
OMP_NUM_THREADS = 2

# --------------------------------------------------------------------------


def intrinsic_memory_for(task: str) -> str:
    return f"intrinsicmemory-{task}"


def intrinsic_arms(task: str) -> list[str]:
    return [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]


def every_arm(task: str) -> list[str]:
    return BASELINE_MEMORIES + intrinsic_arms(task)


def account_directive() -> str:
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def vllm_serve_block() -> str:
    return f"""cd {VLLM_VENV_DIR}

source .venv/bin/activate

YAML_CONFIG="{YAML_CONFIG}"
HF_HOME={HF_HOME}
MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/{MODEL_SNAPSHOT}/
MODEL_NAME="{MODEL_NAME}"

export TIKTOKEN_ENCODINGS_BASE="{TIKTOKEN_ENCODINGS_BASE}"

srun \\
    --nodes=$SLURM_NNODES \\
    --gpus=$SLURM_GPUS \\
    --cpus-per-task {CPUS_PER_TASK} \\
    --ntasks-per-node 1 \\
    vllm serve $MODEL_PATH \\
    --served-model-name $MODEL_NAME \\
    --config $YAML_CONFIG \\
    --host 0.0.0.0 \\
    --port {PORT} \\
    --max-num-seqs {MAX_NUM_SEQS} \\
    --max-num-batched-tokens {MAX_NUM_BATCHED_TOKENS} \\
    --max-model-len {MAX_MODEL_LEN} \\
    {"--enable-prefix-caching" if ENABLE_PREFIX_CACHING else "--no-enable-prefix-caching"} \\
    --tensor_parallel_size={TENSOR_PARALLEL_SIZE} &

VLLM_PID=$!

# wait for vllm to start up
until curl -s http://localhost:{PORT}/health > /dev/null 2>&1; do
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"
curl -s http://localhost:{PORT}/v1/models

deactivate"""


def run_command(task: str, memories: list[str], cross_task: bool,
                background: bool = False, seeds: list[int] = SEEDS, scope: str = "") -> str:
    """One `tasks/run.py` invocation, for one dataset and its arms.

    A backgrounded one records its own pid: a bare `wait` would also wait on the
    vLLM server started the same way, which only ever exits when killed.

    `--resume` makes a job that is submitted again finish what the last one ran
    out of time for. It skips whichever experiments already have a row in
    ${DB_DIR}, so pointing a job at a directory whose results you meant to
    replace will skip them rather than redo them - use a new DB_DIR for that.
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


def preamble(job_name: str, output_pattern: str, script_name: str, db_dir: str,
             time_limit: str = TIME_LIMIT, scratch_dir: str = SCRATCH_DIR) -> str:
    """Everything before the run: the allocation, the server, the environment."""
    return f"""#!/bin/bash
{account_directive()}#SBATCH --job-name={job_name}
#SBATCH --nodes={NODES}
#SBATCH --gpus={GPUS}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={output_pattern}

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

# Every job of one experiment set must point at the same directory: they append to
# one overall_results.csv under a lock on the file. A new sweep needs a new one:
#   DB_DIR={PROJECT_DIR}/results/sweep-next sbatch slurm/{script_name}
DB_DIR=${{DB_DIR:-{db_dir}}}

{vllm_serve_block()}

# experiment setup
export MODEL_NAME="{MODEL_NAME}"
export OPENAI_API_BASE=http://localhost:{PORT}/v1
export OPENAI_API_KEY="none"
export OMP_NUM_THREADS={OMP_NUM_THREADS}

cd {REPO_DIR}
source .venv/bin/activate

sleep {VLLM_STARTUP_SLEEP}

# The experiment processes only, not vLLM: vLLM keeps the node-local TMPDIR it
# started with, which is faster and big enough for one process.
export TMPDIR="${{SCRATCH_DIR:-{scratch_dir}}}/{job_name}-${{SLURM_JOB_ID}}"
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


def render_experiment(task: str, sweep: str) -> str:
    return (
        preamble(f"vllm-{task}", f"{log_dir_for(sweep)}/{task}-%x.%j.%t.out",
                 f"{task}_experiment.sh", db_dir=db_dir_for(sweep))
        + "\n"
        + run_command(task, every_arm(task), cross_task=False)
        + CLEANUP
    )


def render_crosstask(sweep: str) -> str:
    """Every dataset's cross-task arms, in one job against one server.

    The datasets get a `run.py` each rather than one sweep over all of them:
    the sweep is a Cartesian product, so a single call would pair every dataset
    with every other dataset's hand-written template. They run concurrently and
    append to the same files under the lock that already makes two submitted
    jobs safe.
    """
    runs = "\n\n".join(
        run_command(task, intrinsic_arms(task), cross_task=True, background=True)
        for task in TASKS
    )

    return (
        preamble("vllm-crosstask", f"{log_dir_for(sweep)}/crosstask-%x.%j.%t.out",
                 "crosstask.sh", db_dir=db_dir_for(sweep))
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


def write_script(script_name: str, body: str) -> None:
    path = SLURM_DIR / script_name
    path.write_text(body)
    path.chmod(0o755)
    print(f"wrote {path}")


def ensure_log_dir(log_dir: str) -> None:
    """Slurm opens the output file before the job's script runs, so it cannot mkdir its own."""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"could not create {log_dir}: {error}", file=sys.stderr)
        print("create it on the cluster before sbatch, or slurm drops the job's output",
              file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        default=DEFAULT_SWEEP,
        help="the sweep these scripts belong to, dating its results and logs. Pass the"
             " existing date when regenerating for a sweep already under way, or its jobs"
             f" write somewhere --resume cannot see (default: today, {DEFAULT_SWEEP})",
    )
    return parser.parse_args()


def main() -> None:
    sweep = parse_args().sweep
    ensure_log_dir(log_dir_for(sweep))
    print(f"sweep {sweep}: results -> {db_dir_for(sweep)}, logs -> {log_dir_for(sweep)}")

    scripts = {f"{task}_experiment.sh": render_experiment(task, sweep) for task in TASKS}
    scripts["crosstask.sh"] = render_crosstask(sweep)

    for script_name, body in scripts.items():
        write_script(script_name, body)


if __name__ == "__main__":
    main()
