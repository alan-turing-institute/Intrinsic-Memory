#!/usr/bin/env python3
"""Generate the slurm scripts a sweep is submitted from.

The generated scripts are not committed; this generator is. They go to
slurm/generated/<model>/, one directory per model, so two models' scripts can
sit side by side - the results file tells their rows apart by the model column.

    uv run slurm/generate_slurm.py                            # every script, gpt-oss, today
    uv run slurm/generate_slurm.py --model qwen3.6-35b-a3b
    uv run slurm/generate_slurm.py --sweep 2026-09             # rejoin a sweep
    uv run slurm/generate_slurm.py smoke --model qwen3.6-35b-a3b
    uv run slurm/generate_slurm.py calibration --task jericho --max_tasks 5

The matrix and the cluster are in slurm/config.py, the models in
slurm/models.py, and the script bodies in slurm/render.py.

Set SLURM_ACCOUNT to name an account in the generated scripts; without it they
submit under the user's default.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

from config import (
    CALIBRATE_TIME_LIMIT,
    GENERATED_DIR,
    SEEDS,
    TASKS,
    db_dir_for,
    log_dir_for,
)
from models import DEFAULT_MODEL, MODELS, Model
from render import render_calibration, render_crosstask, render_experiment, render_smoke

# The sweep a script belongs to, dating its results and logs so a new one lands
# beside the last rather than on top of it: a run refuses to append to a results
# file whose header is not its schema.
#
# Today by default, and `--sweep` to name an existing one. Pass it whenever
# regenerating scripts for a sweep already under way - every job of a sweep has
# to name the same results directory for `--resume` to see what the last one
# finished, and a sweep outlives the day its scripts were generated on.
DEFAULT_SWEEP = date.today().isoformat()

# --- calibration ---------------------------------------------------------

CALIBRATION_SEEDS = SEEDS[:1]
CALIBRATION_MAX_TASKS = 20

# Jericho's prompt tokens grow with the square of its 100-trial budget, so 20
# tasks would be ~288M tokens - an 18-hour job. Five at 20 trials is ~8M.
CALIBRATION_OVERRIDES = {"jericho": {"max_tasks": 5, "max_trials": 20}}


def scope_flags(task: str, max_tasks: int | None, max_trials: int | None) -> str:
    """The --max_tasks/--max_trials a calibration of `task` runs at.

    A flag given on the command line wins over CALIBRATION_OVERRIDES, for every
    dataset.
    """
    overrides = CALIBRATION_OVERRIDES.get(task, {})
    if max_tasks is None:
        max_tasks = overrides.get("max_tasks", CALIBRATION_MAX_TASKS)
    if max_trials is None:
        max_trials = overrides.get("max_trials")

    flags = f"\n\t--max_tasks {max_tasks} \\"
    if max_trials is not None:
        flags += f"\n\t--max_trials {max_trials} \\"
    return flags


def calibration_db_dir(sweep: str) -> str:
    return f"{db_dir_for(sweep)}-calibration"


# --- writing -------------------------------------------------------------


def write_script(model: Model, script_name: str, body: str) -> Path:
    directory = GENERATED_DIR / model.slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / script_name
    path.write_text(body)
    path.chmod(0o755)
    print(f"wrote {path}")
    return path


def ensure_log_dir(log_dir: str) -> None:
    """Slurm opens the output file before the job's script runs, so it cannot mkdir its own."""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"could not create {log_dir}: {error}", file=sys.stderr)
        print("create it on the cluster before sbatch, or slurm drops the job's output",
              file=sys.stderr)


# --- command line --------------------------------------------------------


def add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """--model, --sweep and --task, accepted either side of the subcommand.

    A subparser repeating an option has to default to SUPPRESS: an ordinary
    default would overwrite the value the main parser had already parsed, so
    `--model qwen3.6-35b-a3b smoke` would silently generate for gpt-oss.
    """
    parser.add_argument(
        "--model", choices=sorted(MODELS),
        default=argparse.SUPPRESS if suppress else DEFAULT_MODEL,
        help=f"which model the scripts serve and ask for (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sweep", default=argparse.SUPPRESS if suppress else DEFAULT_SWEEP,
        help="the sweep these scripts belong to, dating its results and logs. Pass the"
             " existing date when regenerating for a sweep already under way, or its jobs"
             f" write somewhere --resume cannot see (default: today, {DEFAULT_SWEEP})",
    )
    parser.add_argument(
        "--task", nargs="+", choices=TASKS,
        default=argparse.SUPPRESS if suppress else TASKS,
        help="the datasets to generate for (default: all of them)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(parser, suppress=False)

    kinds = parser.add_subparsers(dest="kind")
    for name, help_text in (
        ("experiment", "one script per dataset, every arm x every seed"),
        ("crosstask", "every dataset's intrinsic arms, cross-task, in one job"),
        ("smoke", "the whole path, small - and how a model is brought up"),
        ("all", "every kind except calibration (the default)"),
    ):
        add_common(kinds.add_parser(name, help=help_text), suppress=True)

    calibrate = kinds.add_parser("calibration", help="short runs that size the real jobs")
    add_common(calibrate, suppress=True)
    calibrate.add_argument(
        "--seed", nargs="+", type=int, default=CALIBRATION_SEEDS,
        help=f"the seeds each arm runs (default: {' '.join(str(s) for s in CALIBRATION_SEEDS)})",
    )
    calibrate.add_argument(
        "--max_tasks", type=int,
        help=f"tasks of the dataset per arm (default: {CALIBRATION_MAX_TASKS}, or the"
             " dataset's entry in CALIBRATION_OVERRIDES)",
    )
    calibrate.add_argument(
        "--max_trials", type=int,
        help="trials per task, overriding the dataset's own budget (default: the budget,"
             " or the dataset's entry in CALIBRATION_OVERRIDES)",
    )
    calibrate.add_argument(
        "--time_limit", default=CALIBRATE_TIME_LIMIT,
        help=f"the #SBATCH --time each job asks for (default: {CALIBRATE_TIME_LIMIT})",
    )
    calibrate.add_argument(
        "--db_dir",
        help="where the runs write, unless DB_DIR is set at submit time (default: the"
             " sweep's results directory with -calibration appended)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MODELS[args.model]
    sweep = args.sweep
    kind = args.kind or "all"

    ensure_log_dir(log_dir_for(sweep))
    print(f"{model.slug}, sweep {sweep}: results -> {db_dir_for(sweep)},"
          f" logs -> {log_dir_for(sweep)}")

    if kind == "calibration":
        db_dir = args.db_dir or calibration_db_dir(sweep)
        print(f"calibration -> {db_dir}")
        for task in args.task:
            write_script(model, f"{task}_calibrate.sh", render_calibration(
                model, task, sweep, seeds=args.seed, db_dir=db_dir,
                scope=scope_flags(task, args.max_tasks, args.max_trials),
                time_limit=args.time_limit,
            ))
        return

    if kind in ("experiment", "all"):
        for task in args.task:
            write_script(model, f"{task}_experiment.sh",
                         render_experiment(model, task, sweep))

    if kind in ("crosstask", "all"):
        write_script(model, "crosstask.sh", render_crosstask(model, sweep, tasks=args.task))

    if kind in ("smoke", "all"):
        write_script(model, "smoke_test.sh", render_smoke(model, sweep))


if __name__ == "__main__":
    main()
