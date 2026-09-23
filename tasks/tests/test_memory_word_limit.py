"""A configured word limit reaches the prompt that rewrites the memory.

Nothing else bounds how much an intrinsic memory module writes, and the module
that generates its own template is the one whose documents vary most, so a limit
that is configured but never reaches the model is indistinguishable from no limit
at all. Asserted per registry key: a module added to `MAS_MEMORY_MODULES` that
takes its own path through `summarize` fails here rather than going uncovered.
"""

import tempfile
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mas.llm import TokenTracker
from mas.memory.mas_memory.prompt import MEMORY_WORD_LIMIT
from mas.module_map import MAS_MEMORY_MODULES
from tasks import results

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM
from tasks.tests.test_intrinsic_tokens import INTRINSIC_KEYS, TRAJECTORY


def update_prompt(key: str, **global_config) -> str:
    """The user message of the memory-update call a module made."""
    llm = FakeLLM(replies=['what the agent learned'], tracker=TokenTracker())
    memory = MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1, **global_config},
        llm_model=llm,
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context('put a mug in the cabinet', task_description='a description')
    for action, observation in TRAJECTORY:
        memory.move_memory_state(action, observation)
    memory.summarize(solver_message='the solver said this')

    assert llm.calls, f'{key} made no LLM call to take an update prompt from'
    return llm.calls[-1][-1].content


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_an_unset_limit_leaves_the_memory_unbounded(key):
    """The limit is opt-in: the arms measured so far ran without one."""
    assert MEMORY_WORD_LIMIT.split('{')[0] not in update_prompt(key), (
        f'{key} bounded its memory with no limit configured'
    )


@pytest.mark.parametrize('limit', [50, 200])
@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_configured_limit_is_the_one_asked_for(key, limit):
    """A limit that arrives as some other number is worse than none at all."""
    prompt = update_prompt(key, memory_word_limit=limit)

    assert MEMORY_WORD_LIMIT.format(word_limit=limit) in prompt, (
        f'{key} did not carry the configured limit of {limit} words into its update prompt'
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_limit_is_stated_alongside_the_memory_it_bounds(key):
    """Stated before the trajectory, the instruction is many thousands of tokens
    from the generation it governs on a long episode."""
    prompt = update_prompt(key, memory_word_limit=200)

    assert prompt.index(MEMORY_WORD_LIMIT.format(word_limit=200)) > prompt.index(
        '## Current Memory'
    ), f'{key} states the word limit before the memory rather than beside it'


def test_two_limits_are_two_experiments():
    """`--resume` skips an experiment whose key a results file already carries, so a
    limit outside the key makes every arm after the first a silent no-op."""
    assert 'memory_word_limit' in results.IDENTITY_COLUMNS, (
        'the word limit is not part of what identifies an experiment'
    )

    arm = {'model': 'm', 'task': 'alfworld', 'mas_type': 'autogen',
           'mas_memory': 'intrinsicmemory-alfworld', 'use_validator': False,
           'intrinsic_cross_task': False, 'seed': 0}

    assert (results.experiment_key({**arm, 'memory_word_limit': 100})
            != results.experiment_key({**arm, 'memory_word_limit': 200})), (
        'two word limits key as the same experiment, so the second would be skipped'
    )


@pytest.mark.parametrize('limit', [None, 100])
def test_a_finished_experiment_keys_as_the_row_it_wrote(tmp_path, limit):
    """`--resume` matches a config against rows read back from a CSV. An unset limit
    leaves the config holding None and the file holding a blank, and an arm whose two
    spellings disagree is rerun on every resubmission however often it finishes."""
    fields = {'model': 'm', 'task': 'alfworld', 'mas_type': 'autogen',
              'mas_memory': 'intrinsicmemory-alfworld', 'use_validator': False,
              'intrinsic_cross_task': False}
    path = str(tmp_path / 'overall_results.csv')

    results.write_row(path, results.AGGREGATE_COLUMNS, {
        **results.identity(**fields, memory_word_limit=limit),
        'max_trials': 30,
        **asdict(results.Measurements.of(
            SimpleNamespace(mean_reward=0.0, mean_done=0.0, mean_trials=0.0, episode_count=1),
            TokenTracker(),
        )),
        'seed': 0,
    })

    key = results.experiment_key({**fields, 'memory_word_limit': limit, 'seed': 0})

    assert key in results.recorded_experiments(path), (
        f'an experiment at limit {limit} does not key as the row it wrote, so it would rerun'
    )
