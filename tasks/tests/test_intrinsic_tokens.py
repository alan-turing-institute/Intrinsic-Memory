"""Which LLM calls are billed to the intrinsic_* token counters.

These two counters are how this fork's cost question is answered - what the
memory spends against the run total - so both directions matter. An intrinsic
module's own update call must land in them, and another memory module's must
not, or the two numbers no longer separate the memory from the agents.
"""

import tempfile

import pytest

from mas.llm import TokenTracker
from mas.module_map import MAS_MEMORY_MODULES

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM

INTRINSIC_KEYS = sorted(key for key in MAS_MEMORY_MODULES if key.startswith('intrinsicmemory'))

# summarize only calls out once the trajectory has some history behind it.
TRAJECTORY = [('look at desk 1', 'you see a mug'), ('take mug', 'you pick up the mug')]


def build(key: str) -> tuple:
    tracker = TokenTracker()
    memory = MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
        llm_model=FakeLLM(tracker=tracker),
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context('put a mug in the cabinet', task_description='a description')
    for action, observation in TRAJECTORY:
        memory.move_memory_state(action, observation)
    return memory, tracker


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_a_memory_update_is_billed_as_intrinsic(key):
    memory, tracker = build(key)

    memory.summarize(solver_message='the solver said this')

    assert tracker.intrinsic_prompt_tokens > 0, f"{key} spent nothing intrinsic on its own update"
    assert tracker.intrinsic_completion_tokens > 0, f"{key} spent nothing intrinsic on its own update"


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_intrinsic_share_is_part_of_the_run_total_not_beside_it(key):
    """A caller reads intrinsic_* as a share of the totals, not an extra column."""
    memory, tracker = build(key)

    memory.summarize()

    assert tracker.intrinsic_prompt_tokens <= tracker.prompt_tokens, (
        f"{key} billed more intrinsic prompt tokens than the run spent in total"
    )
    assert tracker.intrinsic_completion_tokens <= tracker.completion_tokens, (
        f"{key} billed more intrinsic completion tokens than the run spent in total"
    )


def test_another_memory_module_bills_nothing_intrinsic():
    """Only the intrinsic family's calls count, so the two numbers stay comparable."""
    memory, tracker = build('chatdev')
    memory.counter = 9  # ChatDev calls out on every tenth summarize

    memory.summarize()

    assert tracker.prompt_tokens > 0, "the fake did not record the call at all"
    assert (tracker.intrinsic_prompt_tokens, tracker.intrinsic_completion_tokens) == (0, 0), (
        "a non-intrinsic memory module's summary was billed to the intrinsic counters"
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_each_update_is_counted_beside_what_it_spent(key):
    """A total over an unknown number of updates cannot say how big one memory is.

    Dividing by the count is the only way to tell a memory that obeyed a length
    limit from a shorter episode that never tested it.
    """
    memory, tracker = build(key)

    memory.summarize()
    memory.summarize()

    assert tracker.intrinsic_updates == 2, (
        f'{key} counted {tracker.intrinsic_updates} updates for two summarize calls'
    )


def test_another_memory_module_counts_no_updates():
    """The count divides the intrinsic token totals, so it takes the same calls."""
    memory, tracker = build('chatdev')
    memory.counter = 9

    memory.summarize()

    assert tracker.intrinsic_updates == 0, (
        "a non-intrinsic memory module's summary was counted as an intrinsic update"
    )
