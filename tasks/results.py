"""The result rows an experiment writes, and the CSV files they go in.

Different results files:
    <task>-<memory>-task_results.csv      one completed task, raw
    <task>-<memory>-seed_<n>-progress.csv one completed task, as means so far
    overall_results.csv                   finished experiment
    failed_tasks.csv                      tasks that could not be run
    failed_experiments.csv                experiments that could not be run

The progress file is a crash-recovery file: it is removed once the experiment is finished.

Two things worth knowing about the token columns:
1. On a task row they are that episode's spend, and they will not sum to the run total when a task failed part-way, because a task with no row still spent what it spent
2. On an aggregate row they are the run's cumulative total.
"""

import csv
import fcntl
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

from mas.llm import TokenTracker

if TYPE_CHECKING:
    from mas.mas import EpisodeResult
    from tasks.envs.base_env import AggregateResults


def _tokens(tracker: TokenTracker) -> dict:
    return {
        'completion_tokens': tracker.completion_tokens,
        'prompt_tokens': tracker.prompt_tokens,
        'intrinsic_completion_tokens': tracker.intrinsic_completion_tokens,
        'intrinsic_prompt_tokens': tracker.intrinsic_prompt_tokens,
    }


@dataclass(frozen=True)
class Measurements:
    """Means over episodes, for the files reporting an aggregate.

    `tasks_scored` is how many episodes the means are over. Without it a reader
    cannot tell a mean over 200 tasks from a mean over 3, which is also how a
    partial progress row reports its progress.
    """

    mean_reward: float
    mean_done: float
    mean_trials: float
    tasks_scored: int
    completion_tokens: int
    prompt_tokens: int
    intrinsic_completion_tokens: int
    intrinsic_prompt_tokens: int

    @classmethod
    def of(cls, averages: "AggregateResults", tracker: TokenTracker) -> "Measurements":
        return cls(
            mean_reward=averages.mean_reward,
            mean_done=averages.mean_done,
            mean_trials=averages.mean_trials,
            tasks_scored=averages.episode_count,
            **_tokens(tracker),
        )


@dataclass(frozen=True)
class TaskMeasurements:
    """One episode, unaggregated.

    `trials` is empty for an episode cut short because an agent could not act -
    how many turns that task needed was never established, and a 0 would read as
    a task that took no turns.
    """

    reward: float
    done: bool
    trials: int
    completion_tokens: int
    prompt_tokens: int
    intrinsic_completion_tokens: int
    intrinsic_prompt_tokens: int

    @classmethod
    def of(cls, episode: "EpisodeResult", spent: TokenTracker) -> "TaskMeasurements":
        return cls(
            reward=episode.reward,
            done=episode.done,
            trials=episode.trials,
            **_tokens(spent),
        )


# Which experiment a row belongs to.
IDENTITY_COLUMNS: tuple[str, ...] = (
    'model', 'task', 'mas_type', 'mas_memory', 'use_validator', 'intrinsic_cross_task'
)

MEASUREMENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(Measurements))
TASK_MEASUREMENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(TaskMeasurements))

# The token counts, in the order they appear wherever they appear.
TOKEN_COLUMNS: tuple[str, ...] = tuple(_tokens(TokenTracker()))

AGGREGATE_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('max_trials',) + MEASUREMENT_COLUMNS + ('seed',)
)

TASK_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('max_trials', 'task_id') + TASK_MEASUREMENT_COLUMNS + ('seed',)
)

FAILED_TASK_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('task_id', 'error_type', 'error_message', 'seed')
)

FAILED_EXPERIMENT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('error_type', 'error_message', 'seed')
)

# What makes two rows the same experiment, for --resume.
KEY_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + ('seed',)

# Default filenames, can be overridden in args config.
TASK_RESULTS_FILENAME = '{task}-{mas_memory}-task_results.csv'
PROGRESS_FILENAME = '{task}-{mas_memory}-seed_{seed}-progress.csv'


def task_results_path(
    working_dir: str, task: str, mas_memory: str, filename: str = TASK_RESULTS_FILENAME
) -> str:
    return os.path.join(working_dir, filename.format(task=task, mas_memory=mas_memory))


def progress_path(
    working_dir: str, task: str, mas_memory: str, seed: int, filename: str = PROGRESS_FILENAME
) -> str:
    return os.path.join(
        working_dir, filename.format(task=task, mas_memory=mas_memory, seed=seed)
    )


def overall_results_path(db_dir: str, filename: str) -> str:
    return os.path.join(db_dir, filename)


def failed_tasks_path(working_dir: str, filename: str) -> str:
    return os.path.join(working_dir, filename)


def failed_experiments_path(db_dir: str, filename: str) -> str:
    return os.path.join(db_dir, filename)


def identity(
    *,
    model: str,
    task: str,
    mas_type: str,
    mas_memory: str,
    use_validator: bool,
    intrinsic_cross_task: bool,
) -> dict:
    return {
        'model': model,
        'task': task,
        'mas_type': mas_type,
        'mas_memory': mas_memory,
        'use_validator': use_validator,
        'intrinsic_cross_task': intrinsic_cross_task,
    }


def aggregate_row(
    *,
    identity_fields: dict,
    seed: int,
    max_trials: int,
    averages: "AggregateResults",
    tracker: TokenTracker,
) -> dict:
    """One row for the progress file or for overall_results.csv.

    `max_trials` is the budget an episode actually got, not the flag: the flag is
    None when the task's configured max_steps applies.
    """
    return {
        **identity_fields,
        'max_trials': max_trials,
        **asdict(Measurements.of(averages, tracker)),
        'seed': seed,
    }


def task_row(
    *,
    identity_fields: dict,
    seed: int,
    max_trials: int,
    task_id: int,
    episode: "EpisodeResult",
    spent: TokenTracker,
) -> dict:
    """One row for the task results file: what one episode did, unaggregated.

    `spent` is what that episode cost, not the run's running total.
    """
    return {
        **identity_fields,
        'max_trials': max_trials,
        'task_id': task_id,
        **asdict(TaskMeasurements.of(episode, spent)),
        'seed': seed,
    }

def recorded_experiments(path: str) -> set[tuple[str, ...]]:
    """The key of every experiment already in a results file."""
    if not os.path.exists(path):
        return set()

    with open(path, newline='', encoding='utf-8') as reader:
        return {
            tuple(str(row[column]) for column in KEY_COLUMNS)
            for row in csv.DictReader(reader)
        }


def remove_progress(path: str) -> None:
    """Drop the crash-recovery file, once the result it was protecting is written."""
    if os.path.exists(path):
        os.remove(path)


def failure_fields(error: Exception) -> dict:
    """A newline would break the row for a reader that does not parse quotes."""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error).replace('\n', ' '),
    }


class SchemaMismatch(RuntimeError):
    """An existing results file's header is not the schema being written.

    Appending anyway is what turns a campaign's CSV into something nobody can
    read, and it cannot be undone from the file afterwards.
    """


def check_header(path: str, columns: tuple[str, ...]) -> None:
    """Raise if `path` already exists under a different schema.

    So a sweep pointed at an existing --db_dir says so before it runs rather than
    when its first experiment finishes.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, encoding='utf-8') as reader:
        existing = reader.readline().rstrip('\n')

    if existing != ','.join(columns):
        raise _mismatch(path, existing, columns)


def _mismatch(path: str, existing: str, columns: tuple[str, ...]) -> SchemaMismatch:
    return SchemaMismatch(
        f'{path} was written by another schema.\n'
        f'  its header: {existing}\n'
        f'  this run:   {",".join(columns)}'
    )


def write_row(path: str, columns: tuple[str, ...], row: dict) -> str:
    """Append one row, writing the header first if the file is new.

    Reading the header and appending are one operation, under a lock on the file
    itself: an experiment set is submitted as several Slurm jobs pointed at one
    --db_dir, and processes in different jobs share nothing else to lock in.

    Returns the line written. Every column must be present in `row`: a row that
    does not fill its schema raises here rather than reaching the file short.
    """
    line = _format(columns, row)
    header = ','.join(columns)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, 'a+', encoding='utf-8') as handle:
        with _locked(handle, path):
            handle.seek(0)
            existing = handle.readline().rstrip('\n')
            if not existing:
                handle.write(header + '\n')
            elif existing != header:
                raise _mismatch(path, existing, columns)
            handle.write(line)
            # Flushed inside the lock: the next writer reads the header back.
            handle.flush()

    return line.rstrip('\n')


_UNLOCKABLE: set[str] = set()


@contextmanager
def _locked(handle, path: str):
    """An exclusive lock on the open file, if the filesystem grants one.

    Lustre supports flock only when mounted with it, and the alternative to
    appending unlocked is refusing to record a finished experiment - so a refusal
    is reported once per file and the write goes ahead.
    """
    locked = False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
    except OSError as error:
        if path not in _UNLOCKABLE:
            _UNLOCKABLE.add(path)
            print(
                f'cannot lock {path} ({error}); concurrent jobs may interleave rows',
                file=sys.stderr,
            )

    try:
        yield
    finally:
        if locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _format(columns: tuple[str, ...], row: dict) -> str:
    missing = set(columns) - set(row)
    if missing:
        raise KeyError(f'row is missing columns {sorted(missing)}')

    buffer = io.StringIO()
    csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n').writerow(row)
    return buffer.getvalue()
