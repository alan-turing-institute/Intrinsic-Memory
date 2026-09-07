"""Contract tests across the registries.

Each family - workflows, recorders, memory modules - has one contract its callers
rely on, asserted here once over the whole family rather than per implementation.

These read the registries (`MAS`, `RECORDERS`, `ENVS`) rather than hard-coded
lists, so a new implementation is covered the moment it is registered.
"""

import numbers
import tempfile

import pytest

from mas.agents import Env
from mas.mas import EpisodeResult
from mas.memory import MASMemoryBase
from mas.module_map import MAS_MEMORY_MODULES, module_map

from tasks.envs import ENVS, RECORDERS
from tasks.envs.base_env import AggregateResults, BaseEnv, aggregate

from tasks.mas_workflow import MAS
from tasks.prompts import get_dataset_system_prompt
from tasks.tests.fakes import (
    FakeEmbeddingFunc,
    FakeEnv,
    FakeLLM,
    RecordingObserver,
    fake_reasoning,
)

MEMORY_CONFIG_KEYS = ("successful_topk", "failed_topk", "insights_topk", "threshold")

# One task config per task, carrying whichever keys that task's recorder reads
# out of current_task_config in task_end.
TASK_CONFIGS = {
    "alfworld": {"env_kwargs": {"gamefile": "/data/pick_and_place_simple-1/game.tw-pw"}},
    "babyai": {},
    "fever": {},
    "hotpotqa": {},
    "jericho": {"id": "detective"},
    "pddl": {"game_name": "blockworld"},
    "sciworld": {},
    "swebench": {"id": "django__django-16485", "repo": "django/django"},
}


def build_workflow(mas_type: str, env: FakeEnv, memory_cls=MASMemoryBase, replies=None,
                   use_projector: bool = False, use_validator: bool = False):
    """Assemble one workflow the way run.py's build_mas does, with fakes."""
    workflow = MAS[mas_type]()
    workflow.add_observer(RecordingObserver())

    llm = FakeLLM(replies=replies)
    memory = memory_cls(
        namespace=f"{mas_type}-contract",
        global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=llm,
        embedding_func=FakeEmbeddingFunc(),
    )
    workflow.build_system(
        fake_reasoning(replies=replies),
        memory,
        env,
        {key: 1 for key in MEMORY_CONFIG_KEYS}
        | {"use_projector": use_projector, "use_validator": use_validator},
    )
    return workflow


# ── D2 · all four workflows return one contract ───────────────────────────────

@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_returns_the_episode_result_run_py_unpacks(mas_type):
    """schedule() must return what run.py reads: reward, done and trials."""
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert isinstance(result, EpisodeResult), (
        f"{mas_type}.schedule() returned {type(result).__name__}, expected EpisodeResult"
    )
    assert isinstance(result.reward, numbers.Real), f"{mas_type}: reward is {result.reward!r}"
    assert isinstance(result.done, bool), f"{mas_type}: done is {result.done!r}"
    assert isinstance(result.trials, int), f"{mas_type}: trials is {result.trials!r}"


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_result_still_unpacks_positionally(mas_type):
    """`reward, done, trials = schedule(...)` must keep working.

    Call sites outside this repo unpack the result positionally.
    """
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env)

    reward, done, trials = workflow.schedule(
        {"task_main": "m", "task_description": "d", "few_shots": []}
    )

    assert (reward, done, trials) == tuple(
        workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})
    )


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_rejects_a_task_config_missing_its_required_keys(mas_type):
    env = FakeEnv()
    workflow = build_workflow(mas_type, env)

    with pytest.raises(ValueError, match="task_main|task_description"):
        workflow.schedule({})


# ── D1 · all four recorders share one signature ───────────────────────────────

@pytest.fixture
def working_dir(tmp_path):
    return str(tmp_path)


def build_recorder(task: str, working_dir: str):
    return RECORDERS[task](working_dir=working_dir, namespace=f"{task}-contract")


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_task_end_accepts_reward_done_and_trials(task, working_dir):
    """run.py hands every recorder the EpisodeResult a workflow returned."""
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))

    recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))

    assert recorder.episodes == [EpisodeResult(reward=1.0, done=True, trials=7)], (
        f"{task}: task_end did not reach BaseRecorder, episodes are {recorder.episodes}"
    )


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_returns_three_numerics(task, working_dir):
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))
    recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))

    results = recorder.average_results()

    assert isinstance(results, AggregateResults), (
        f"{task}: average_results() returned {type(results).__name__}"
    )
    assert all(isinstance(value, numbers.Real) for value in results), f"{task}: {results!r}"


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_works_before_any_task_has_ended(task, working_dir):
    """A sweep over an empty task list must report zeros, not divide by zero."""
    recorder = build_recorder(task, working_dir)

    assert recorder.average_results() == (0, 0, 0, 0)


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_task_end_before_task_begin_is_rejected(task, working_dir):
    """The base class refuses to record a result it cannot attribute to a task."""
    recorder = build_recorder(task, working_dir)

    with pytest.raises(RuntimeError, match="task id or the task config"):
        recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))


# ── D4 · every memory module works with every workflow ────────────────────────

# `g-memory` persists through langchain_chroma, which conftest stubs with a
# MagicMock - far enough to import, not far enough to drive an episode. Tracked in
# docs/BACKLOG.md; every other registered module is exercised below.
UNTESTABLE_OFFLINE = {"g-memory"}

# Read from the registry, so a newly registered module is covered without editing
# this file - it either runs here or has to be named in UNTESTABLE_OFFLINE.
MEMORY_KEYS = sorted(set(MAS_MEMORY_MODULES) - UNTESTABLE_OFFLINE)


def test_the_offline_exclusions_all_still_exist():
    """A module removed from the registry must not stay on the exclusion list."""
    stale = UNTESTABLE_OFFLINE - set(MAS_MEMORY_MODULES)

    assert not stale, f"UNTESTABLE_OFFLINE names {sorted(stale)}, which is not registered"


@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("memory_key", MEMORY_KEYS)
def test_every_memory_module_survives_every_workflow(mas_type, memory_key):
    """Every memory module must survive an episode driven by any workflow.

    Mostly this exercises the summarize() keyword contract, which is the one
    place the workflows and the memory modules have to agree.
    """
    _, memory_cls = module_map("io", memory_key)
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env, memory_cls=memory_cls)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert isinstance(result, EpisodeResult)


# ── the trials convention ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("steps_to_done, expected_trials", [(1, 1), (2, 2), (3, 3)])
def test_trials_counts_the_trials_completed(mas_type, steps_to_done, expected_trials):
    """trials is a count of completed trials, not a loop index."""
    env = FakeEnv(max_trials=5, steps_to_done=steps_to_done)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert result.done is True
    assert result.trials == expected_trials
    assert result.trials == len(env.actions), "trials should match the steps actually taken"


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_an_unsolved_episode_reports_its_whole_budget(mas_type):
    env = FakeEnv(max_trials=4, steps_to_done=99)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert result.done is False
    assert result.trials == 4


# ── what the aggregate is a mean over ─────────────────────────────────────────

def test_aggregate_averages_over_the_episodes_it_is_given():
    episodes = [
        EpisodeResult(reward=1.0, done=True, trials=2),
        EpisodeResult(reward=0.0, done=False, trials=8),
    ]

    assert aggregate(episodes) == AggregateResults(
        mean_reward=0.5, mean_done=0.5, mean_trials=5.0, episode_count=2
    )


def test_an_episode_with_no_trial_count_still_counts_towards_reward_and_done():
    """An aborted episode was not solved, so it belongs in the success rate. What
    it cannot contribute to is the average number of turns a task takes."""
    episodes = [
        EpisodeResult(reward=1.0, done=True, trials=4),
        EpisodeResult(reward=0.0, done=False, trials=None),
    ]

    averages = aggregate(episodes)

    assert averages.mean_done == 0.5, "the unsolved episode must pull the rate down"
    assert averages.episode_count == 2
    assert averages.mean_trials == 4.0, "the mean turns is over the one episode that reported"


def test_an_aggregate_of_nothing_is_zeroes_not_a_division_error():
    assert aggregate([]) == AggregateResults(0, 0, 0, 0)


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_episode_count_reports_the_denominator(task, working_dir):
    """The averages hide their denominator otherwise: a mean over 120 of 134
    tasks is not comparable to one over all 134."""
    recorder = build_recorder(task, working_dir)
    for task_id in range(3):
        recorder.task_begin(task_id, dict(TASK_CONFIGS[task]))
        recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=2))

    assert recorder.average_results().episode_count == 3


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_the_recorder_keeps_the_episodes_it_was_given(task, working_dir):
    """One list of episodes, rather than parallel lists per field that have to be
    kept in step."""
    recorder = build_recorder(task, working_dir)
    given = [
        EpisodeResult(reward=1.0, done=True, trials=2),
        EpisodeResult(reward=0.0, done=False, trials=None),
    ]
    for task_id, episode in enumerate(given):
        recorder.task_begin(task_id, dict(TASK_CONFIGS[task]))
        recorder.task_end(episode)

    assert recorder.episodes == given


# ── the environment interface ─────────────────────────────────────────────────

def test_the_offline_fake_satisfies_the_env_protocol():
    """What licenses the rest of this suite: the fake answers the real interface.

    `Env` is structural, so FakeEnv does not inherit it and nothing else would
    notice the two drifting apart.
    """
    assert isinstance(FakeEnv(), Env)


@pytest.mark.parametrize("task", sorted(ENVS))
def test_every_registered_environment_is_a_base_env(task):
    """BaseEnv is where the trial budget and the abstract methods are enforced."""
    assert issubclass(ENVS[task], BaseEnv), f"{task} would not inherit __init__ or the ABC checks"


# ── every env classifies its own reasoning steps ──────────────────────────────

# One reasoning step and one real action per task, in that task's own vocabulary.
# The families spell a reasoning step two different ways, which is why the
# workflow has to ask the environment rather than match a substring itself.
THOUGHT_VOCABULARY = {
    "alfworld": ("think: I need to find a mug.", "go to shelf 1"),
    "babyai": ("think: the red ball is off to my left.", "turn left"),
    "jericho": ("think: the chief handed me a sheet of paper.", "take paper"),
    "sciworld": ("think: I should heat the water.", "focus on the thermometer"),
    "pddl": ("think: I should stack the blocks.", "stack block_a block_b"),
    "fever": ("Thought 1: I need to search Telemundo.", "Search[Telemundo]"),
    "hotpotqa": ("Thought 1: I need to search Milhouse.", "Lookup[named after]"),
    "swebench": ("think: the traceback names defaultfilters.py.", "ls tests"),
}


@pytest.mark.parametrize("task", sorted(ENVS))
def test_every_env_knows_a_reasoning_step_from_an_action(task):
    """`_solver_stuck` asks the env this, so a wrong answer is a loop it never breaks."""
    thought, action = THOUGHT_VOCABULARY[task]

    assert ENVS[task].is_thought(thought) is True, f"{task} calls {thought!r} an action"
    assert ENVS[task].is_thought(action) is False, f"{task} calls {action!r} a thought"


def _respellings(thought: str) -> list[str]:
    """The same reasoning step as a model variously writes it.

    A model asked for `think:` supplies `Think:` and `THINK:`, and copies the
    `> ` marker out of the few shots along with it.
    """
    marker, _, rest = thought.partition(":")

    return [
        thought,
        f"{marker.capitalize()}:{rest}",
        f"{marker.upper()}:{rest}",
        f"> {thought}",
        f"**{thought}**",
    ]


@pytest.mark.parametrize("task", sorted(ENVS))
def test_a_reasoning_step_is_recognised_however_the_model_capitalises_it(task):
    """Case cost a trial on every dataset: a thought the env does not recognise is
    sent to the simulator as an action, and rejected.

    Asserted per registry key rather than per environment, so no dataset can be
    added or changed back into the strict form without this failing.
    """
    thought, _ = THOUGHT_VOCABULARY[task]

    for spelling in _respellings(thought):
        assert ENVS[task].is_thought(spelling) is True, (
            f"{task} would send {spelling!r} to the simulator as an action"
        )


def test_pddl_accepts_the_second_form_its_own_prompt_offers():
    """PDDL's instructions give `think xxx:` alongside `think: xxx`, so a model
    following them writes the marker without a colon after it.

    Jericho is the one environment that cannot allow this, because `think` is a
    verb its parser accepts as a command - so it asks for the colon and the rest
    do not.
    """
    assert ENVS["pddl"].is_thought("think I should move block A first:") is True
    assert ENVS["pddl"].is_thought("think") is True

    assert ENVS["jericho"].is_thought("think") is False, (
        "interactive fiction takes `think` as a command, so it cannot be a marker"
    )


# How a model dresses up the same action. Each has to survive to the same command.
DECORATIONS = ('{}', '"{}"', "'{}'", '{}.', '1. {}', '- {}', '**{}**', '```\n{}\n```')


@pytest.mark.parametrize("task", sorted(ENVS))
@pytest.mark.parametrize("decoration", DECORATIONS)
def test_an_action_survives_the_way_a_model_formats_it(task, decoration):
    """Each of these forms cost a trial, on every dataset."""
    _, action = THOUGHT_VOCABULARY[task]
    plain = ENVS[task].process_action(action)

    assert ENVS[task].process_action(decoration.format(action)) == plain, (
        f"{task} loses the action out of {decoration.format(action)!r}"
    )


@pytest.mark.parametrize("task", sorted(ENVS))
def test_an_action_written_after_a_thought_is_the_one_taken(task):
    """A reply carrying both can only be done one way round, and taking the
    thought spends the trial achieving nothing - the shape issue #31 described.
    """
    thought, action = THOUGHT_VOCABULARY[task]

    taken = ENVS[task].process_action(f"{thought}\n{action}")

    assert taken == ENVS[task].process_action(action), f"{task} took the thought, not {action!r}"
    assert not ENVS[task].is_thought(taken)


# ── every arm is told the same thing about the shape of a reply ───────────────

# Wordings that constrain the shape of a reply rather than its content. A memory
# module that introduces one of these is telling its agent something the arms it
# is measured against were not told.
FORMAT_INSTRUCTIONS = (
    "one action",
    "single line",
    "one line",
    "nothing else",
)


@pytest.mark.parametrize("task", sorted(ENVS))
def test_every_dataset_asks_for_one_action_per_reply(task):
    """Otherwise the stop sequence is the only thing asking, and it gets dropped.

    `GPTChat` drops `stop=['\\n']` for any endpoint it truncates, which is every
    reasoning model, so a prompt that never asks for one line is relying on
    something that will not be there.
    """
    prompt = get_dataset_system_prompt(task, task_config=dict(TASK_CONFIGS[task]))

    assert any(phrase in prompt for phrase in FORMAT_INSTRUCTIONS), (
        f"{task}'s system prompt never asks for a single action"
    )


@pytest.mark.parametrize("memory_key", MEMORY_KEYS)
def test_no_memory_module_adds_a_format_instruction_the_others_lack(memory_key, tmp_path):
    """The memory module is the variable under test, so it cannot also be the thing
    that decides how parseable the agent's replies are.

    `IntrinsicMASMemory` used to append "You can only perform one action. Output in
    a single line your next action" in the task description it renders, and no other
    module did - so the intrinsic arms were the only ones asked for one line.
    Measured on llama3.1:8b with the stop sequence dropped, that instruction takes
    replies needing repair from 12 of 16 down to 3 of 16, which is a difference in
    prompts showing up as a difference between arms.
    """
    _, memory_cls = module_map("io", memory_key)
    memory = memory_cls(
        namespace=f"{memory_key}-format",
        global_config={"working_dir": str(tmp_path), "hop": 1},
        llm_model=FakeLLM(),
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context("task", "a description")

    summary = memory.summarize(solver_message="sys")

    offending = [phrase for phrase in FORMAT_INSTRUCTIONS if phrase in summary]
    assert not offending, (
        f"{memory_key} tells its agent {offending}, which the other arms are not told"
    )
