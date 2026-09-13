"""Picking a dataset up where a killed job left it.

An experiment's tasks run in one sequential pass, and a job that runs out of
wall clock part-way through leaves its finished tasks in the task results file
and no row in the overall results. Resuming reads those tasks back: it skips
what is already scored and folds it into the means the finished experiment
reports, so two passes report what one uninterrupted pass would have.
"""

from pathlib import Path

from mas.mas import EpisodeResult

from tasks.tests.fakes import FakeEnv
from tasks.tests.test_run_task import StubMAS, build_manager, read_csv

import results


IDENTITY = {
    "model": "fake-model",
    "task": "fever",
    "mas_type": "autogen",
    "mas_memory": "empty",
    "use_validator": False,
    "intrinsic_cross_task": False,
}


def write_task_rows(path: Path, rows: list[dict]) -> None:
    for row in rows:
        results.write_row(str(path), results.TASK_COLUMNS, row)


def task_row(task_id: int, *, seed: int = 42, reward: float = 1.0, done: bool = True,
             trials: int = 1, tokens: int = 100, **identity) -> dict:
    fields = {**IDENTITY, **identity}
    return results.task_row(
        identity_fields=fields,
        seed=seed,
        max_trials=30,
        task_id=task_id,
        episode=EpisodeResult(reward=reward, done=done, trials=trials),
        spent=results.TokenTracker(
            completion_tokens=tokens,
            prompt_tokens=tokens,
            intrinsic_completion_tokens=0,
            intrinsic_prompt_tokens=0,
        ),
    )


# ── reading the partial results back ──────────────────────────────────────────

def test_no_file_means_nothing_has_been_scored(tmp_path):
    key = results.experiment_key({**IDENTITY, "seed": 42})

    assert results.completed_tasks(str(tmp_path / "absent.csv"), key) == {}


def test_the_tasks_already_scored_are_reported_with_their_measurements(tmp_path):
    path = tmp_path / "fever-empty-task_results.csv"
    write_task_rows(path, [
        task_row(0, reward=1.0, done=True, trials=2),
        task_row(1, reward=0.0, done=False, trials=5),
    ])
    key = results.experiment_key({**IDENTITY, "seed": 42})

    completed = results.completed_tasks(str(path), key)

    assert sorted(completed) == [0, 1]
    assert completed[0].reward == 1.0 and completed[0].done is True
    assert completed[1].trials == 5, "a resumed task keeps the trial count it took"


def test_an_episode_that_established_no_trial_count_keeps_reporting_none(tmp_path):
    """A 0 here would read as a task that took no turns and skew mean_trials."""
    path = tmp_path / "fever-empty-task_results.csv"
    write_task_rows(path, [task_row(0, trials=None)])
    key = results.experiment_key({**IDENTITY, "seed": 42})

    assert results.completed_tasks(str(path), key)[0].trials is None


def test_another_seeds_rows_are_not_this_experiments(tmp_path):
    """One task results file holds every seed of an arm, keyed by the row."""
    path = tmp_path / "fever-empty-task_results.csv"
    write_task_rows(path, [task_row(0, seed=42), task_row(1, seed=99)])
    key = results.experiment_key({**IDENTITY, "seed": 42})

    assert sorted(results.completed_tasks(str(path), key)) == [0]


def test_the_same_arm_carrying_memory_forward_is_a_different_experiment(tmp_path):
    path = tmp_path / "fever-empty-task_results.csv"
    write_task_rows(path, [
        task_row(0, intrinsic_cross_task=False),
        task_row(1, intrinsic_cross_task=True),
    ])
    key = results.experiment_key({**IDENTITY, "seed": 42, "intrinsic_cross_task": True})

    assert sorted(results.completed_tasks(str(path), key)) == [1]


# ── resuming a dataset ────────────────────────────────────────────────────────

def test_a_task_already_scored_is_not_run_again(run_task_module, tmp_path):
    run = run_task_module
    write_task_rows(tmp_path / "fever-empty-task_results.csv", [task_row(0), task_row(1)])
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv', resume=True)

    assert len(mas.scheduled) == 2, "only the tasks with no row should have been run"
    assert [config["task"] for config in mas.scheduled] == ["claim 2", "claim 3"]


def test_without_resume_a_scored_task_runs_again(run_task_module, tmp_path):
    """Resuming is opt-in: the default stays one pass over the whole dataset."""
    run = run_task_module
    write_task_rows(tmp_path / "fever-empty-task_results.csv", [task_row(0), task_row(1)])
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv')

    assert len(mas.scheduled) == 4


def test_the_means_span_the_resumed_tasks_and_the_new_ones(run_task_module, tmp_path):
    """Two passes have to report what one uninterrupted pass would have."""
    run = run_task_module
    write_task_rows(tmp_path / "fever-empty-task_results.csv", [
        task_row(0, reward=1.0, done=True, trials=1),
        task_row(1, reward=1.0, done=True, trials=1),
    ])
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(FakeEnv(), outcomes=[EpisodeResult(reward=0.0, done=False, trials=3)])
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv', resume=True)

    averages = manager.recorder.average_results()
    assert averages.episode_count == 4, "the means cover the whole dataset, not the new half"
    assert averages.mean_reward == 0.5
    assert averages.mean_trials == 2.0


def test_the_resumed_tasks_tokens_are_still_charged_to_the_experiment(run_task_module, tmp_path):
    """The token totals are the experiment's, so a resumed pass cannot undercount."""
    run = run_task_module
    write_task_rows(tmp_path / "fever-empty-task_results.csv", [
        task_row(0, tokens=100), task_row(1, tokens=100),
    ])
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv', resume=True)

    assert manager.token_tracker.completion_tokens >= 200
    assert manager.token_tracker.prompt_tokens >= 200


def test_a_resumed_task_is_not_written_a_second_time(run_task_module, tmp_path):
    """One task, one row: a duplicate would be counted twice by any later pass."""
    run = run_task_module
    path = tmp_path / "fever-empty-task_results.csv"
    write_task_rows(path, [task_row(0), task_row(1)])
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv', resume=True)

    ids = [row["task_id"] for row in read_csv(path)]
    assert ids == ["0", "1", "2"], "the resumed rows stay as they were, with no duplicate"


def test_a_dataset_already_finished_schedules_nothing(run_task_module, tmp_path):
    run = run_task_module
    write_task_rows(tmp_path / "fever-empty-task_results.csv",
                    [task_row(i) for i in range(3)])
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, working_dir=str(tmp_path),
                 failed_tasks_filename='failed_tasks.csv', resume=True)

    assert mas.scheduled == []
    assert manager.recorder.average_results().episode_count == 3
