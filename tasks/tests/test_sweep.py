"""Which experiments a sweep runs, and how they are kept apart.

Three properties, all of which fail invisibly rather than loudly:

  - a flag given several values must produce one experiment per combination;
  - every experiment must reach a worker exactly once;
  - two seeds of one config must not share a memory persistence directory.

The last one is the expensive kind of wrong: memory carried from one seed into
another does not fail, it just quietly answers a different question.
"""

import importlib
import sys
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.envs.base_env import BaseRecorder
from tasks.tests.fakes import FakeEnv
from tasks.tests.test_run_task import StubMAS, read_csv


@pytest.fixture
def sweep_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("sweep")


@pytest.fixture
def experiment_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("experiment")


def parse(sweep, *argv: str):
    return sweep.build_arg_parser().parse_args(list(argv))


# ── C3 · one experiment per combination ───────────────────────────────────────

def test_every_combination_of_the_swept_flags_is_one_experiment(sweep_module):
    sweep = sweep_module
    args = parse(
        sweep,
        '--mas_type', 'autogen',
        '--task', 'fever', 'pddl',
        '--mas_memory', 'empty', 'voyager', 'g-memory',
        '--seed', '1', '2', '3',
    )

    configs = sweep.build_experiment_configs(args)

    assert len(configs) == 2 * 3 * 3, "2 tasks x 3 memory modules x 3 seeds"
    distinct = {(config.task, config.mas_memory, config.seed) for config in configs}
    assert len(distinct) == len(configs), "the same experiment appears more than once"


def test_one_value_per_flag_is_one_experiment(sweep_module):
    sweep = sweep_module
    args = parse(sweep, '--mas_type', 'autogen', '--task', 'fever', '--mas_memory', 'empty')

    assert len(sweep.build_experiment_configs(args)) == 1


def test_new_parser_flag_requires_an_experiment_config_field(sweep_module):
    args = parse(
        sweep_module, '--mas_type', 'autogen', '--task', 'fever', '--mas_memory', 'empty'
    )
    args.untracked_flag = 'would otherwise be silently accepted'

    with pytest.raises(TypeError, match='unexpected: untracked_flag'):
        sweep_module.build_experiment_configs(args)


# ── the LLM flags reach the process that makes the calls ─────────────────────

def test_the_llm_flags_are_the_request_an_experiment_makes(
    sweep_module, experiment_module
):
    """A worker is spawned, so it installs its own settings from its own config."""
    from mas.llm import GPTChat, Message
    from tasks.tests.fakes import chat_over_fake_completions

    args = parse(
        sweep_module, '--mas_type', 'autogen', '--task', 'fever', '--mas_memory', 'empty',
        '--max_tokens', '4096', '--temperature', '0.9', '--request_timeout', '15',
    )
    config = sweep_module.build_experiment_configs(args)[0]

    experiment_module.install_llm_settings(config)

    chat, completions = chat_over_fake_completions(['go to desk 1'])
    chat([Message('user', 'what next?')])

    assert completions.calls[0]['max_completion_tokens'] == 4096, (
        f'the token budget did not reach the request: {completions.calls[0]}'
    )
    assert completions.calls[0]['temperature'] == 0.9
    assert GPTChat(model_name='fake-model').client.timeout == 15.0, (
        'the timeout is the client\'s, which GPTChat builds for itself'
    )


def test_the_mas_loggers_only_reach_stderr_when_the_run_asked_for_it(
    sweep_module, experiment_module
):
    import logging

    from mas.logging_utils import CONSOLE_HANDLER_NAME

    def installed(*flags) -> bool:
        for handler in list(logging.getLogger('mas').handlers):
            if handler.name == CONSOLE_HANDLER_NAME:
                logging.getLogger('mas').removeHandler(handler)
        args = parse(
            sweep_module, '--mas_type', 'autogen', '--task', 'fever',
            '--mas_memory', 'empty', *flags,
        )
        experiment_module.install_llm_settings(sweep_module.build_experiment_configs(args)[0])
        return any(
            handler.name == CONSOLE_HANDLER_NAME
            for handler in logging.getLogger('mas').handlers
        )

    try:
        assert not installed(), 'a sweep of 100,000 requests would echo every one'
        assert installed('--log_responses')
    finally:
        for handler in list(logging.getLogger('mas').handlers):
            if handler.name == CONSOLE_HANDLER_NAME:
                logging.getLogger('mas').removeHandler(handler)


# ── C1 · running them, in this process or in a pool ───────────────────────────

class FakeExecutor:
    """ProcessPoolExecutor's shape, running each submission inline."""

    def __init__(self, log: list, max_workers=None, mp_context=None):
        self.log = log
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, function, *args):
        self.log.append(args)
        future = Future()
        future.set_result(function(*args))
        return future


def test_a_single_experiment_does_not_spawn_a_pool(sweep_module, monkeypatch):
    sweep = sweep_module
    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(sweep, 'ProcessPoolExecutor', _refuse_to_be_used)

    assert sweep.run_experiments([{'seed': 42}], num_workers=8) == [{'ran': 42}]


def test_one_worker_runs_every_experiment_in_this_process(sweep_module, monkeypatch):
    sweep = sweep_module
    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(sweep, 'ProcessPoolExecutor', _refuse_to_be_used)

    outcomes = sweep.run_experiments([{'seed': 1}, {'seed': 2}], num_workers=1)

    assert outcomes == [{'ran': 1}, {'ran': 2}]


def test_every_experiment_reaches_a_worker_exactly_once(sweep_module, monkeypatch):
    sweep = sweep_module
    submissions: list[tuple] = []

    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(
        sweep, 'multiprocessing', SimpleNamespace(get_context=lambda method: method)
    )
    monkeypatch.setattr(
        sweep, 'ProcessPoolExecutor', lambda **kwargs: FakeExecutor(submissions, **kwargs)
    )

    experiments = [{'seed': seed} for seed in (1, 2, 3)]
    outcomes = sweep.run_experiments(experiments, num_workers=2)

    assert [config for (config,) in submissions] == experiments
    assert sorted(outcome['ran'] for outcome in outcomes) == [1, 2, 3]


def test_no_more_workers_are_started_than_there_are_experiments(sweep_module, monkeypatch):
    sweep = sweep_module
    started: list = []

    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {})
    monkeypatch.setattr(
        sweep, 'multiprocessing', SimpleNamespace(get_context=lambda method: method)
    )
    monkeypatch.setattr(
        sweep, 'ProcessPoolExecutor', lambda **kwargs: FakeExecutor(started, **kwargs)
    )

    sweep.run_experiments([{'seed': 1}, {'seed': 2}], num_workers=32)

    assert len(started) == 2


def _refuse_to_be_used(*args, **kwargs):
    raise AssertionError("a worker pool was spawned for work that runs in this process")


# ── C2 · two seeds of one config are kept apart ───────────────────────────────

def build_config(tmp_path, seed: int, *extra_args: str):
    """A real parser-derived config, so test fixtures cannot drift from the CLI."""
    sweep = importlib.import_module("sweep")
    args = parse(
        sweep,
        '--mas_type', 'autogen',
        '--task', 'fever',
        '--mas_memory', 'empty',
        '--model', 'fake-model',
        '--max_trials', '3',
        '--max_tokens', '512',
        '--seed', str(seed),
        '--num_workers', '1',
        '--db_dir', str(tmp_path),
        *extra_args,
    )
    return sweep.build_experiment_configs(args)[0]


def stub_build_task(experiment, tasks: list[dict] = ()):
    """build_task's signature, over a fake env and workflow."""
    def build(task, mas_type, memory_type, seed, working_dir, model=None, max_trials=None,
              max_tasks=None, env_overrides=None):
        env = FakeEnv(max_trials=max_trials or 3)
        return experiment.TaskManager(
            task_name=task,
            mas_type=mas_type,
            memory_type=memory_type,
            tasks=[dict(config) for config in tasks],
            env=env,
            recorder=BaseRecorder(working_dir=working_dir, namespace=f'seed_{seed}'),
            mas=StubMAS(env),
            seed=seed,
            model=model,
            token_tracker=experiment.TokenTracker(),
        )
    return build


def test_two_seeds_of_one_config_do_not_share_a_memory_directory(
    experiment_module, monkeypatch, tmp_path
):
    experiment = experiment_module
    memory_dirs: list[str] = []

    monkeypatch.setattr(experiment, 'build_task', stub_build_task(experiment))
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment,
        'run_task',
        lambda manager, working_dir, failed_tasks_filename: memory_dirs.append(
            manager.mem_config['working_dir']
        ),
    )

    for seed in (42, 43):
        outcome = experiment.run_experiment(build_config(tmp_path, seed))
        assert outcome['status'] == 'success', outcome.get('error')

    assert len(set(memory_dirs)) == 2, (
        f"both seeds persisted their memory to the same place: {memory_dirs}"
    )


def test_a_worker_installs_its_settings_before_it_builds_anything(
    experiment_module, monkeypatch, tmp_path
):
    """Its config is all a spawned worker gets, and a GPTChat has to have settings."""
    from mas.settings import default_llm_settings, reset_default_llm_settings

    experiment = experiment_module
    monkeypatch.setattr(experiment, 'build_task', stub_build_task(experiment))
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, 'run_task', lambda *args, **kwargs: None)
    reset_default_llm_settings()

    outcome = experiment.run_experiment(replace(build_config(tmp_path, seed=42), max_tokens=4096))

    assert outcome['status'] == 'success', outcome.get('error')
    assert default_llm_settings().max_tokens == 4096


def test_a_flag_that_configures_an_llm_call_reaches_its_settings(
    sweep_module, experiment_module, monkeypatch, tmp_path
):
    """The route from --max_tokens_ceiling to the budget a retry may climb to."""
    from mas.settings import default_llm_settings, reset_default_llm_settings

    experiment = experiment_module
    monkeypatch.setattr(experiment, 'build_task', stub_build_task(experiment))
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, 'run_task', lambda *args, **kwargs: None)
    reset_default_llm_settings()
    outcome = experiment.run_experiment(
        build_config(tmp_path, 42, '--max_tokens_ceiling', '3333')
    )

    assert outcome['status'] == 'success', outcome.get('error')
    assert default_llm_settings().max_tokens_ceiling == 3333


def test_a_flag_that_configures_an_environment_reaches_it(
    sweep_module, experiment_module, monkeypatch, tmp_path
):
    """A flag declared and then not plumbed through changes nothing and says nothing.

    The environment's own settings ride in the env_config its task's YAML
    supplies, which is where every environment already reads its configuration.
    """
    experiment = experiment_module
    built: dict = {}

    def fake_get_env(task, config, max_trials):
        built.update(config)
        return FakeEnv(max_trials=max_trials)

    def fake_build_mas(manager, *args, **kwargs):
        manager.token_tracker = experiment.TokenTracker()

    monkeypatch.setattr(experiment, 'get_env', fake_get_env)
    monkeypatch.setattr(experiment, 'get_task', lambda task, **kwargs: [{'task': 'a task'}])
    monkeypatch.setattr(experiment, 'build_mas', fake_build_mas)
    monkeypatch.setattr(experiment, 'run_task', lambda *args, **kwargs: None)
    outcome = experiment.run_experiment(
        build_config(
            tmp_path, 42, '--wikipedia_attempts', '7', '--unreachable_search_limit', '9'
        )
    )

    assert outcome['status'] == 'success', outcome.get('error')
    assert built['wikipedia_attempts'] == 7, built
    assert built['unreachable_search_limit'] == 9, built


# ── the progress file is a backup, and lives exactly as long as it is needed ───

def config_dir(tmp_path) -> Path:
    """Where run_experiment puts one config's files, from its own path scheme."""
    return Path(tmp_path) / 'fake-model' / 'fever' / 'autogen' / 'empty'


def test_the_progress_file_is_removed_once_the_experiment_has_its_result(
    experiment_module, monkeypatch, tmp_path
):
    """Written by run_task, deleted by run_experiment - so the two must agree on the path."""
    experiment = experiment_module
    monkeypatch.setattr(experiment, 'CONFIG', {'fever': {'few_shots_num': 1}})
    monkeypatch.setattr(experiment, 'get_task_few_shots', lambda **kwargs: ['a few shot'])
    monkeypatch.setattr(experiment, 'get_dataset_system_prompt', lambda *a, **k: 'do the task')
    monkeypatch.setattr(
        experiment, 'build_task', stub_build_task(experiment, [{'task': 'a'}, {'task': 'b'}])
    )
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)

    outcome = experiment.run_experiment(build_config(tmp_path, seed=42))
    assert outcome['status'] == 'success', outcome.get('error')

    written = sorted(path.name for path in config_dir(tmp_path).glob('*.csv'))
    assert written == ['fever-empty-task_results.csv'], (
        f"the progress file should not have survived the experiment: {written}"
    )
    assert len(read_csv(config_dir(tmp_path) / 'fever-empty-task_results.csv')) == 2, (
        "the raw task rows are what the progress file was standing in for, and are kept"
    )


def test_the_progress_file_survives_an_experiment_that_failed(
    experiment_module, monkeypatch, tmp_path
):
    """The case it exists for: no result was written, so the partial one is all there is."""
    experiment = experiment_module

    def run_task_then_fail(task_manager, working_dir, failed_tasks_filename):
        experiment.results.write_row(
            experiment.results.progress_path(
                working_dir, task_manager.task_name, task_manager.memory_type, task_manager.seed
            ),
            experiment.results.AGGREGATE_COLUMNS,
            experiment.results.aggregate_row(
                identity_fields=task_manager.identity(),
                seed=task_manager.seed,
                max_trials=task_manager.env.max_trials,
                averages=task_manager.recorder.average_results(),
                tracker=task_manager.token_tracker,
            ),
        )
        raise RuntimeError('the endpoint went away mid-dataset')

    monkeypatch.setattr(experiment, 'build_task', stub_build_task(experiment))
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, 'run_task', run_task_then_fail)

    outcome = experiment.run_experiment(build_config(tmp_path, seed=42))

    assert outcome['status'] == 'failed'
    assert (config_dir(tmp_path) / 'fever-empty-seed_42-progress.csv').exists(), (
        "deleting the backup of a run that failed is the one thing this must not do"
    )
    assert len(read_csv(Path(tmp_path) / 'failed_experiments.csv')) == 1


# ── a killed sweep can be continued rather than repeated ──────────────────────

def run_one_experiment(experiment, monkeypatch, config) -> None:
    """One real run_experiment over a fake env and workflow, writing real rows."""
    monkeypatch.setattr(experiment, 'CONFIG', {'fever': {'few_shots_num': 1}})
    monkeypatch.setattr(experiment, 'get_task_few_shots', lambda **kwargs: ['a few shot'])
    monkeypatch.setattr(experiment, 'get_dataset_system_prompt', lambda *a, **k: 'do the task')
    monkeypatch.setattr(
        experiment, 'build_task', stub_build_task(experiment, [{'task': 'a'}])
    )
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)

    outcome = experiment.run_experiment(config)
    assert outcome['status'] == 'success', outcome.get('error')


def test_an_experiment_that_finished_is_not_run_again(
    sweep_module, experiment_module, monkeypatch, tmp_path
):
    """The key has to match between a config and the row that config produced."""
    config = build_config(tmp_path, seed=42)
    run_one_experiment(experiment_module, monkeypatch, config)

    remaining, already_done = sweep_module.experiments_to_run(
        [config], str(Path(tmp_path) / 'overall_results.csv')
    )

    assert (remaining, already_done) == ([], 1)


def test_another_seed_of_the_same_config_is_still_run(
    sweep_module, experiment_module, monkeypatch, tmp_path
):
    run_one_experiment(experiment_module, monkeypatch, build_config(tmp_path, seed=42))
    other_seed = build_config(tmp_path, seed=43)

    remaining, already_done = sweep_module.experiments_to_run(
        [other_seed], str(Path(tmp_path) / 'overall_results.csv')
    )

    assert (remaining, already_done) == ([other_seed], 0)


def test_the_cross_task_arm_is_not_mistaken_for_the_baseline(
    sweep_module, experiment_module, monkeypatch, tmp_path
):
    """The two arms share a mas_memory value; the column is what separates them."""
    run_one_experiment(experiment_module, monkeypatch, build_config(tmp_path, seed=42))
    cross_task = replace(build_config(tmp_path, seed=42), intrinsic_cross_task=True)

    remaining, already_done = sweep_module.experiments_to_run(
        [cross_task], str(Path(tmp_path) / 'overall_results.csv')
    )

    assert (remaining, already_done) == ([cross_task], 0)
