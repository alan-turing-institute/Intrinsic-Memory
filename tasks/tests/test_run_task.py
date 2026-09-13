"""run_task's loop over a dataset.

Drives the real run_task with a fake environment, workflow and LLM.

Failure handling is split in two, and both are pinned here:

  - an agent that cannot produce an action is handled inside schedule, where the
    trial count is known, and the task IS scored, as unsolved;
  - a fault around the episode (set_env, prompt assembly, the recorder) is
    recorded and NOT scored, since it is not an episode outcome.

Either way the next task in the dataset runs.
"""

import csv
from pathlib import Path

import pytest

from mas.mas import EpisodeResult

from tasks.envs import RECORDERS
from tasks.envs.base_env import BaseRecorder
from tasks.tests.fakes import FakeEnv
from tasks.tests.test_task_smoke import SMOKE_TASK_CONFIGS


class StubMAS:
    """Stands in for a built workflow: run_task reaches .env, .agents_team and
    .schedule."""

    def __init__(self, env, outcomes=None, fail_on=()):
        self.env = env
        self.agents_team = {}
        self.outcomes = outcomes
        self.fail_on = set(fail_on)
        self.scheduled: list[dict] = []

    def schedule(self, task_config: dict) -> EpisodeResult:
        index = len(self.scheduled)
        self.scheduled.append(dict(task_config))
        if index in self.fail_on:
            raise RuntimeError(f"workflow exploded on task {index}")
        if self.outcomes:
            return self.outcomes[index % len(self.outcomes)]
        return EpisodeResult(reward=1.0, done=True, trials=1)


class ExplodingEnv(FakeEnv):
    """An environment whose set_env fails for chosen task indices."""

    def __init__(self, fail_on=(), **kwargs):
        super().__init__(**kwargs)
        self.fail_on = set(fail_on)
        self.set_env_calls = 0

    def set_env(self, task_config: dict) -> tuple[str, str]:
        index = self.set_env_calls
        self.set_env_calls += 1
        if index in self.fail_on:
            raise RuntimeError(f"simulator unavailable for task {index}")
        return super().set_env(task_config)


def build_manager(run, tmp_path, tasks, mas, task_name="fever", recorder=None, seed=42):
    """`fever` is the default only because its task config is the smallest; the
    loop and its failure handling do not read anything task-specific."""
    if recorder is None:
        recorder = BaseRecorder(working_dir=str(tmp_path), namespace="run-task-test")
    return run.TaskManager(
        task_name=task_name,
        mas_type="autogen",
        memory_type="empty",
        tasks=tasks,
        env=mas.env,
        recorder=recorder,
        mas=mas,
        seed=seed,
        model="fake-model",
        token_tracker=run.TokenTracker(),
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as reader:
        return list(csv.DictReader(reader))


# ── the happy path ────────────────────────────────────────────────────────────

def test_every_task_in_the_dataset_is_scheduled(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}", "answer": "SUPPORTS"} for i in range(4)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert len(mas.scheduled) == 4
    assert [e.reward for e in manager.recorder.episodes] == [1.0] * 4


def test_every_task_gets_a_row_of_its_own_raw_numbers(run_task_module, tmp_path):
    """The means are recoverable from the task rows; the reverse is not true."""
    run = run_task_module
    outcomes = [
        EpisodeResult(reward=1.0, done=True, trials=1),
        EpisodeResult(reward=0.0, done=False, trials=3),
        EpisodeResult(reward=0.0, done=False, trials=None),
    ]
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv(), outcomes=outcomes)
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    rows = read_csv(tmp_path / "fever-empty-task_results.csv")
    assert [row["task_id"] for row in rows] == ["0", "1", "2"]
    assert [row["reward"] for row in rows] == ["1.0", "0.0", "0.0"]
    assert [row["done"] for row in rows] == ["True", "False", "False"]
    assert [row["trials"] for row in rows] == ["1", "3", ""], (
        "an episode that never established a trial count reports no number, not zero"
    )


def test_progress_is_checkpointed_after_every_completed_task(run_task_module, tmp_path):
    """A job killed part-way through a dataset leaves a readable result."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas, seed=7)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    rows = read_csv(tmp_path / "fever-empty-seed_7-progress.csv")
    assert len(rows) == 3, "one row should be appended per completed task, not one at the end"
    assert [row["tasks_scored"] for row in rows] == ["1", "2", "3"], (
        "each row's means are over the tasks scored so far"
    )


# ── one failing task does not stop the rest ───────────────────────────────────

def test_a_failing_environment_does_not_stop_the_remaining_tasks(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(5)]
    mas = StubMAS(ExplodingEnv(fail_on=(2,)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert len(mas.scheduled) == 4, "the four healthy tasks should all have run"
    assert [e.reward for e in manager.recorder.episodes] == [1.0] * 4


def test_a_failing_workflow_does_not_stop_the_remaining_tasks(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(FakeEnv(), fail_on=(0, 2))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert len(mas.scheduled) == 4, "every task should have been attempted"
    assert [e.reward for e in manager.recorder.episodes] == [1.0, 1.0], "only the two that ran are scored"


def test_a_failed_task_is_recorded_with_its_error(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(1,)))
    manager = build_manager(run, tmp_path, tasks, mas, seed=7)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    rows = read_csv(tmp_path / "failed_tasks.csv")
    assert len(rows) == 1
    assert rows[0]["task_id"] == "1"
    assert rows[0]["error_type"] == "RuntimeError"
    assert "simulator unavailable" in rows[0]["error_message"]
    assert rows[0]["seed"] == "7"


def test_a_failed_task_is_excluded_from_the_averages_not_scored_as_zero(
    run_task_module, tmp_path
):
    """An environment fault is not an episode outcome, so it is not scored."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(0,)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    averages = manager.recorder.average_results()
    assert (averages.mean_reward, averages.mean_done) == (1.0, 1.0), (
        "the mean is over the two tasks that ran"
    )
    assert len(manager.recorder.episodes) == 2


def test_the_exclusion_is_stated_loudly(run_task_module, tmp_path, capsys):
    """results.csv does not record how many tasks are behind its mean."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(ExplodingEnv(fail_on=(0, 3)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert "2/4 tasks failed" in capsys.readouterr().err


def test_a_dataset_where_every_task_fails_still_finishes(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(0, 1, 2)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert manager.recorder.average_results() == (0, 0, 0, 0)
    assert len(read_csv(tmp_path / "failed_tasks.csv")) == 3


# ── the failure handling holds for every task, not just the smallest one ───────

@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_a_failing_task_is_isolated_whatever_the_task(run_task_module, tmp_path, task, monkeypatch):
    """The generic loop is task-agnostic, but the recorders are not.

    Each real recorder reads different keys out of current_task_config and raises
    its own errors - AlfworldRecorder indexes env_kwargs.gamefile, PDDLRecorder
    demands game_name - so the isolation has to be asserted against all four
    rather than inferred from one.
    """
    run = run_task_module
    monkeypatch.setattr(run, "CONFIG", {task: {"few_shots_num": 1}})

    tasks = [dict(SMOKE_TASK_CONFIGS[task]) for _ in range(4)]
    recorder = RECORDERS[task](working_dir=str(tmp_path), namespace=f"{task}-isolation")
    mas = StubMAS(ExplodingEnv(fail_on=(1,)))
    manager = build_manager(run, tmp_path, tasks, mas, task_name=task, recorder=recorder)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert len(mas.scheduled) == 3, f"[{task}] the three healthy tasks should have run"
    assert len(read_csv(tmp_path / "failed_tasks.csv")) == 1
    assert len(recorder.episodes) == 3


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_a_task_config_a_recorder_rejects_is_isolated_too(
    run_task_module, tmp_path, task, monkeypatch
):
    """A malformed task config is a per-task fault like any other.

    This is the failure mode that differs per task: an empty config is fine for
    the FEVER and SciWorld recorders and fatal for the ALFWorld and PDDL ones,
    which is exactly why it should not stop the sweep either way.
    """
    run = run_task_module
    monkeypatch.setattr(run, "CONFIG", {task: {"few_shots_num": 1}})

    tasks = [dict(SMOKE_TASK_CONFIGS[task]), {}, dict(SMOKE_TASK_CONFIGS[task])]
    recorder = RECORDERS[task](working_dir=str(tmp_path), namespace=f"{task}-malformed")
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas, task_name=task, recorder=recorder)

    run.run_task(manager, working_dir=str(tmp_path), failed_tasks_filename='failed_tasks.csv')

    assert len(recorder.episodes) >= 2, (
        f"[{task}] the two well-formed tasks should have been scored"
    )
