"""The experiment matrix and the cluster the sweep runs on.

Edit these and rerun `slurm/generate_slurm.py`. What varies by model is in
`slurm/models.py`; how a script is put together is in `slurm/render.py`.
"""
import os
from pathlib import Path

SLURM_DIR = Path(__file__).parent
# Generated scripts, one directory per model. Not committed.
GENERATED_DIR = SLURM_DIR / "generated"
TEMPLATE_DIR = SLURM_DIR / "templates"

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

# The budget a call starts at, per task; tasks not listed use DEFAULT_MAX_TOKENS.
# A starved reasoning model is retried with a doubled budget, so too small a
# start is paid for in whole wasted calls rather than in a truncated answer.
MAX_TOKENS_OVERRIDES = {"babyai": 4096, "pddl": 4096}
DEFAULT_MAX_TOKENS = 2048

# --- where a job writes --------------------------------------------------
# One user's project allocation, and the root of everything a job writes.
# Home is a 101 G quota; one sweep's results are 17 G, its logs 15 G, and the
# scratch ALFWorld churns through is larger than the quota on its own.
PROJECT_DIR = "/projects/u6vh/syuen.u6vh"

# Where the experiment processes put their temporary files. Not the node-local
# scratch TMPDIR names by default: ALFWorld's PDDL engine copies a 28.8 MB
# libdownward.so into it on every environment load, and 100 experiments doing
# that at once exhausted it - 10,358 tasks of one sweep died on ENOSPC.
SCRATCH_DIR = f"{PROJECT_DIR}/tmp"


def db_dir_for(sweep: str) -> str:
    return f"{PROJECT_DIR}/results/sweep-{sweep}"


def log_dir_for(sweep: str) -> str:
    return f"{PROJECT_DIR}/logs/{sweep}"


# --- cluster -------------------------------------------------------------

REPO_DIR = "~/GMemory"
NODES = 1
GPUS = 4
CPUS_PER_TASK = 16
TIME_LIMIT = "24:00:00"
SMOKE_TIME_LIMIT = "00:30:00"
CALIBRATE_TIME_LIMIT = "02:00:00"
PORT = 8000
VLLM_STARTUP_SLEEP = 100

# Named at generation time rather than committed: it is one user's allocation.
ACCOUNT = os.environ.get("SLURM_ACCOUNT")

# One worker process per experiment, each loading an embedding model on the CPU.
# Unset, every one of them sizes its thread pool to the whole node.
OMP_NUM_THREADS = 2
