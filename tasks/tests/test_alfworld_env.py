"""The ALFWorld environment: what it plays, and what it scores.

The simulator is TextWorld, which the dev environment does not install, so the
batch env `AlfredTWEnv.init_env` returns is faked here. The fake answers the
shapes the real one does - a list of commands in, tuples of one element out,
and the win flag on `info['won']` rather than on `done` - because that is where
the two most easily drift apart, and where the scoring bug below lived.
"""

import copy
import importlib
import json
import sys

import pytest
import yaml

from tasks.envs import ENVS, TASKS_PATH, get_task, alfworld_env
from tasks.envs.alfworld_env import (
    REJECTED,
    get_env_name_from_gamefile,
    prefixes,
)
from tasks.prompts import alfworld_few_shots

from mas.utils import repo_path

with open('tasks/env_configs/alfworld_config.yaml') as reader:
    ENV_CONFIG: dict = yaml.safe_load(reader)

GOAL = (
    'You are in the middle of a room. Looking quickly around you, you see a desk 1, '
    'and a shelf 1.\n\nYour task is to: put a mug in the desk.___0'
)


class FakeBatchEnv:
    """One TextWorld batch of one game, driven by a script of replies.

    A script entry is `(observation, won, done)`. Past the end of the script the
    game answers as it does to a command it did not understand.
    """

    def __init__(self, script: list[tuple[str, bool, bool]]):
        self.script = script
        self.commands: list[str] = []
        self.resets = 0
        self.closed = 0

    def reset(self):
        self.resets += 1
        return ['You are in the middle of a room.'], {'won': [False]}

    def step(self, commands: list[str]):
        self.commands.append(commands[0])
        observation, won, done = self.script.pop(0) if self.script else (REJECTED, False, False)

        return [observation], (0,), (done,), {'won': [won]}

    def close(self):
        self.closed += 1


class FakeAlfredTWEnv:
    """`AlfredTWEnv`'s surface: the games it is told to play, and the batch it builds."""

    def __init__(self, config: dict, train_eval: str = 'train'):
        self.config = config
        self.train_eval = train_eval
        self.game_files: list[str] = []
        self.script: list[tuple[str, bool, bool]] = []
        self.built: list[FakeBatchEnv] = []

    def init_env(self, batch_size: int) -> FakeBatchEnv:
        if not self.game_files:
            raise AssertionError('init_env was called with no game to play')

        self.built.append(FakeBatchEnv(self.script))
        return self.built[-1]


@pytest.fixture
def gamefile(tmp_path):
    path = tmp_path / 'pick_and_place_simple-Mug-None-Desk-308' / 'trial_1' / 'game.tw-pddl'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    return path


@pytest.fixture
def build(monkeypatch):
    """An AlfworldEnv over the shipped config, with the simulator faked."""
    monkeypatch.setattr(alfworld_env, 'get_environment', lambda name: FakeAlfredTWEnv)

    def make(max_trials: int = 30, script: list = None):
        # A deep copy because the constructor writes the trial budget into the
        # config's nested sections, which a shallow one would share between tests.
        env = ENVS['alfworld'](env_config=copy.deepcopy(ENV_CONFIG), max_trials=max_trials)
        env.main_env.script = list(script or [])
        return env

    return make


def task_config(gamefile, **overrides) -> dict:
    return {
        'task': GOAL,
        'env_kwargs': {'config': 'alfworld', 'gamefile': str(gamefile)},
        'task_type': 'put',
        'env_name': 'pick_and_place',
    } | overrides


# ── the registry imports whether or not the simulator is installed ────────────

class _BlockAlfworld:
    """An import hook that makes `alfworld` absent, as it is on a plain `uv sync`."""

    def find_spec(self, name, path=None, target=None):
        if name == 'alfworld' or name.startswith('alfworld.'):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


@pytest.fixture
def without_the_simulator(monkeypatch):
    for name in [name for name in sys.modules if name.split('.')[0] == 'alfworld']:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, 'tasks.envs.alfworld_env', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BlockAlfworld()] + sys.meta_path)

    return importlib.import_module('tasks.envs.alfworld_env')


def test_the_module_imports_with_the_simulator_absent(without_the_simulator):
    """`tasks/envs/__init__.py` imports every environment at module scope, and
    alfworld is the one simulator `uv sync` cannot install - both its compiled
    dependencies lack an aarch64 wheel. Imported eagerly here, its absence is a
    ModuleNotFoundError for babyai, fever, hotpotqa, jericho, pddl and sciworld
    too, on every machine that has not installed it by hand.
    """
    assert without_the_simulator.get_environment is None
    assert without_the_simulator.prefixes, 'the rest of the module still has to load'


def test_building_the_environment_without_the_simulator_says_where_to_get_it(
    without_the_simulator,
):
    """The absence has to surface when alfworld is asked for, not before."""
    with pytest.raises(ModuleNotFoundError, match='data/data.md'):
        without_the_simulator.AlfworldEnv(env_config=copy.deepcopy(ENV_CONFIG), max_trials=30)


# ── nothing is played until a game is chosen ──────────────────────────────────

def test_constructing_the_environment_starts_no_game(build):
    """`get_env` builds one of these per experiment and `set_env` chooses the
    game, so a game started in the constructor is one no task asked for."""
    env = build()

    assert env.env is None, 'a game was started before any task named one'
    assert env.main_env.built == []


def test_reset_before_a_game_is_chosen_is_refused(build):
    """`register_games` on an empty game list builds a batch of no games whose
    `reset` never returns, so the experiment hangs instead of failing."""
    env = build()

    with pytest.raises(RuntimeError, match='set_env'):
        env.reset()


@pytest.mark.parametrize(
    'config',
    [
        {},
        {'env_name': 'pick_and_place'},
        {'env_kwargs': {'gamefile': '/games/game.tw-pddl'}},
        {'env_kwargs': {}, 'env_name': 'pick_and_place'},
    ],
)
def test_a_task_config_missing_the_game_or_its_type_is_rejected(build, config):
    with pytest.raises(ValueError, match='`env_kwargs.gamefile` and an `env_name`'):
        build().set_env(config)


def test_a_missing_game_names_the_file_it_wanted(build):
    """The games are downloaded separately, so this is the first thing to go
    wrong on a fresh checkout and the error has to say what to fetch."""
    env = build()

    with pytest.raises(FileNotFoundError, match='nowhere/game.tw-pddl'):
        env.set_env(task_config('/nowhere/game.tw-pddl'))

    with pytest.raises(FileNotFoundError, match='data/data.md'):
        env.set_env(task_config('/nowhere/game.tw-pddl'))


def test_a_gamefile_from_the_manifest_is_resolved_against_the_repository(build):
    """The manifest's paths are repository-relative, and a job is started from
    wherever the scheduler put it."""
    relative = 'data/alfworld/json_2.1.1/valid_unseen/nothing-here/game.tw-pddl'

    with pytest.raises(FileNotFoundError, match=str(repo_path(relative))):
        build().set_env(task_config(relative))


# ── one game per task ─────────────────────────────────────────────────────────

def test_the_chosen_game_is_the_only_one_played(build, gamefile):
    env = build()

    env.set_env(task_config(gamefile))

    assert env.main_env.game_files == [str(gamefile)]
    assert env.main_env.built[-1].resets == 1


def test_a_second_task_gets_a_fresh_interpreter_and_the_first_is_closed(build, gamefile, tmp_path):
    """One interpreter per game and 134 games per experiment, so an interpreter
    left open is a process left running for the length of the run."""
    second = tmp_path / 'look_at_obj_in_light-Bowl-None-DeskLamp-308' / 'trial_1' / 'game.tw-pddl'
    second.parent.mkdir(parents=True)
    second.write_text('{}')
    env = build()

    env.set_env(task_config(gamefile))
    first_played = env.main_env.built[-1]
    env.set_env(task_config(second, env_name='look_at_obj'))

    assert len(env.main_env.built) == 2, 'the second task replayed the first interpreter'
    assert first_played.closed == 1, 'the first interpreter was left open'


def test_the_task_name_and_description_come_out_of_the_goal(build, gamefile):
    task_main, task_description = build().set_env(task_config(gamefile))

    assert task_main.startswith('pick_and_place-put a mug in the desk')
    assert task_description == GOAL.split('___')[0]
    assert '___' not in task_description, 'the manifest suffix would reach the agent'


# ── the trial budget ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('max_trials', [30, 100])
def test_the_trial_budget_is_the_limit_the_simulator_enforces(build, max_trials):
    """`init_env` takes `max_episode_steps` from whichever of these sections its
    `training_method` names, so a budget above the config's own number is cut
    short by the simulator and the remaining trials achieve nothing."""
    env = build(max_trials=max_trials)

    for section in ('rl', 'dagger'):
        assert env.main_env.config[section]['training']['max_nb_steps_per_episode'] == max_trials, (
            f'{section} would end the episode at its own step limit'
        )


# ── what the episode scored ───────────────────────────────────────────────────

def test_an_episode_that_ran_out_of_steps_is_not_scored_as_a_win(build, gamefile):
    """TextWorld reports `done` when the step limit is reached as well as on a
    win, so an episode scored on `done` records a success for every task that
    merely ran out of turns."""
    env = build(script=[('You arrive at desk 1.', False, True)])
    env.set_env(task_config(gamefile))

    _, _, done = env.step('go to desk 1')

    assert done is True, 'the episode is over either way'
    assert env.feedback() == (0.0, False, 'You failed the task.')


def test_a_win_is_scored(build, gamefile):
    env = build(script=[('You put the mug 1 in/on the desk 1.', True, True)])
    env.set_env(task_config(gamefile))

    observation, reward, done = env.step('put mug 1 in/on desk 1')

    assert (reward, done) == (1, True)
    assert env.feedback() == (1.0, True, 'You successfully finished this task!')
    assert observation == 'You put the mug 1 in/on the desk 1.'


def test_a_command_the_game_refused_is_scored_below_one_it_carried_out(build, gamefile):
    """The agent is told nothing else about a command that did not parse, so the
    reward is the only signal separating it from one that simply did not finish
    the task."""
    env = build(script=[(REJECTED, False, False), ('You arrive at desk 1.', False, False)])
    env.set_env(task_config(gamefile))

    assert env.step('fly to the moon')[1] == -1
    assert env.step('go to desk 1')[1] == 0


def test_a_new_task_starts_from_an_unwon_episode(build, gamefile, tmp_path):
    """`feedback` is read once per episode; a win left set would score the next
    task as solved before it took a turn."""
    second = tmp_path / 'pick_clean_then_place-Mug-None-Desk-1' / 'trial_1' / 'game.tw-pddl'
    second.parent.mkdir(parents=True)
    second.write_text('{}')
    env = build(script=[('You put the mug 1 in/on the desk 1.', True, True)])
    env.set_env(task_config(gamefile))
    env.step('put mug 1 in/on desk 1')

    env.set_env(task_config(second, env_name='pick_clean_then_place'))

    assert env.feedback() == (0.0, False, 'You failed the task.')


# ── a reasoning step is not a command ─────────────────────────────────────────

@pytest.mark.parametrize(
    'thought',
    [
        'think: I need to find a mug.',
        'Think: I need to find a mug.',
        '> think: I need to find a mug.',
    ],
)
def test_a_reasoning_step_never_reaches_the_simulator(build, gamefile, thought):
    """A thought sent as a command is answered `Nothing happens.`, which spends
    the trial and tells the agent nothing."""
    env = build()
    env.set_env(task_config(gamefile))

    observation, _, done = env.step(thought)

    assert observation == 'OK.'
    assert done is False
    assert env.main_env.built[-1].commands == [], 'the thought was sent to the game'


def test_the_action_written_after_a_thought_is_the_one_taken(build, gamefile):
    env = build(script=[('You arrive at shelf 1.', False, False)])
    env.set_env(task_config(gamefile))

    env.step('think: a mug is likely on a shelf.\ngo to shelf 1')

    assert env.main_env.built[-1].commands == ['go to shelf 1']


# ── the manifest ──────────────────────────────────────────────────────────────

def test_every_task_in_the_manifest_names_a_game_and_a_type():
    tasks = get_task('alfworld')

    assert tasks, 'the manifest parsed to nothing'
    assert all(task['env_kwargs']['gamefile'].endswith('game.tw-pddl') for task in tasks)
    assert all(task['env_name'] in prefixes for task in tasks), (
        'a game file no prefix matches is one the recorder cannot group'
    )


def test_every_task_type_in_the_manifest_has_the_few_shots_its_prompt_asks_for():
    """`get_task_few_shots` indexes the shots by task type, so a type without
    them raises before the task has a chance to run."""
    for task in get_task('alfworld'):
        for index in (0, 2):
            key = f'react_{task["task_type"]}_{index}'
            assert key in alfworld_few_shots, f'{task["env_name"]} has no {key}'


def test_the_loader_keeps_each_goal_with_its_own_game():
    """Reversing the two raises nothing: every task would be played on the wrong
    game and scored against another task's goal."""
    with open(TASKS_PATH['alfworld']) as reader:
        source = json.load(reader)

    tasks = get_task('alfworld')

    assert [task['task'] for task in tasks] == [row['goal'] for row in source]
    assert [task['env_kwargs']['gamefile'] for task in tasks] == [
        row['gamefile'] for row in source
    ]
    assert [task['env_name'] for task in tasks] == [
        get_env_name_from_gamefile(row['gamefile']) for row in source
    ]
