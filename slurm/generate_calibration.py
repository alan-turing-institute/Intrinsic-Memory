#!/usr/bin/env python3
"""Generate the per-task slurm/*_calibrate.sh runs that size the sweep jobs.

One dataset, every arm, one seed, twenty tasks at the full trial budget, in a
2-hour allocation. The result rows carry the token spend, so tokens / elapsed =
throughput, and episodes x tokens-per-task / throughput = the wall clock a real
job needs.

    uv run slurm/generate_calibration.py                  # every dataset
    uv run slurm/generate_calibration.py --task fever pddl
    uv run slurm/generate_calibration.py --max_tasks 5 --time_limit 00:30:00

Every default below is a flag, so a calibration can be resized without editing
the file. The cluster, the model and the arms are generate_slurm.py's.
"""
import argparse

from generate_slurm import (
    CLEANUP,
    SEEDS,
    TASKS,
    every_arm,
    preamble,
    run_command,
    write_script,
)

DEFAULT_SEEDS = SEEDS[:1]
DEFAULT_MAX_TASKS = 20
DEFAULT_TIME_LIMIT = "02:00:00"
DEFAULT_DB_DIR = "$HOME/GMemory/.db-calibration"

# Jericho's prompt tokens grow with the square of its 100-trial budget, so 20
# tasks would be ~288M tokens - an 18-hour job. Five at 20 trials is ~8M.
OVERRIDES = {"jericho": {"max_tasks": 5, "max_trials": 20}}

SUMMARY = """

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


def scope_flags(task: str, max_tasks: int | None, max_trials: int | None) -> str:
    """The --max_tasks/--max_trials a calibration of `task` runs at.

    A flag given on the command line wins over OVERRIDES, for every dataset.
    """
    overrides = OVERRIDES.get(task, {})
    if max_tasks is None:
        max_tasks = overrides.get("max_tasks", DEFAULT_MAX_TASKS)
    if max_trials is None:
        max_trials = overrides.get("max_trials")

    flags = f"\n\t--max_tasks {max_tasks} \\"
    if max_trials is not None:
        flags += f"\n\t--max_trials {max_trials} \\"
    return flags


def render(task: str, *, seeds: list[int], time_limit: str, db_dir: str,
           max_tasks: int | None, max_trials: int | None) -> str:
    return (
        preamble(
            f"vllm-{task}-calibrate",
            f"out/{task}-calibrate-%x.%j.%t.out",
            f"{task}_calibrate.sh",
            time_limit=time_limit,
            db_dir=db_dir,
        )
        + "\n"
        + run_command(
            task,
            every_arm(task),
            cross_task=False,
            seeds=seeds,
            scope=scope_flags(task, max_tasks, max_trials),
        )
        + SUMMARY
        + CLEANUP
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        nargs="+",
        choices=TASKS,
        default=TASKS,
        help="the datasets to calibrate (default: all of them)",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"the seeds each arm runs (default: {' '.join(str(s) for s in DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        help=f"tasks of the dataset per arm (default: {DEFAULT_MAX_TASKS}, or the dataset's"
             " entry in OVERRIDES)",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        help="trials per task, overriding the dataset's own budget (default: the budget,"
             " or the dataset's entry in OVERRIDES)",
    )
    parser.add_argument(
        "--time_limit",
        default=DEFAULT_TIME_LIMIT,
        help=f"the #SBATCH --time each job asks for (default: {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--db_dir",
        default=DEFAULT_DB_DIR,
        help=f"where the runs write, unless DB_DIR is set at submit time (default:"
             f" {DEFAULT_DB_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task in args.task:
        write_script(
            f"{task}_calibrate.sh",
            render(
                task,
                seeds=args.seed,
                time_limit=args.time_limit,
                db_dir=args.db_dir,
                max_tasks=args.max_tasks,
                max_trials=args.max_trials,
            ),
        )


if __name__ == "__main__":
    main()
