"""The two controls that separate a hand-written template's structure from its content.

`frozentemplate-<task>` hands the agent the hand-written template's JSON as a
memory that is never updated: the template's domain knowledge, without the
memory mechanism. `intrinsicmemory-<task>-schema` runs the memory mechanism over
the same fields with every value emptied: the structure, without the knowledge.
"""

import json
import re
import tempfile

import pytest

from mas.llm import TokenTracker
from mas.module_map import MAS_MEMORY_MODULES

from tasks.envs import ENVS
from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM

HAND_TASKS = sorted(task for task in ENVS if f'intrinsicmemory-{task}' in MAS_MEMORY_MODULES)
FROZEN_KEYS = sorted(key for key in MAS_MEMORY_MODULES if key.startswith('frozentemplate-'))
SCHEMA_KEYS = sorted(key for key in MAS_MEMORY_MODULES if key.endswith('-schema'))

TRAJECTORY = [('look at desk 1', 'you see a mug'), ('take mug', 'you pick up the mug')]

KEY = re.compile(r'"([^"\n]+)"\s*:')
QUOTED = re.compile(r'"([^"\n]*)"(\s*:)?')


def build(key: str, llm=None):
    memory = MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
        llm_model=llm or FakeLLM(tracker=TokenTracker()),
        embedding_func=FakeEmbeddingFunc(),
    )
    start_task(memory)
    return memory


def start_task(memory, task='put a mug in the cabinet'):
    memory.init_task_context(task, task_description='a description')
    for action, observation in TRAJECTORY:
        memory.move_memory_state(action, observation)


def hand_prompt(task: str) -> str:
    return MAS_MEMORY_MODULES[f'intrinsicmemory-{task}'].system_prompt


def template_block(prompt: str) -> str:
    """From the first line that is a bare `{` to the next line that is a bare `}`."""
    lines = prompt.splitlines()
    start = lines.index('{')
    end = lines.index('}', start)
    return '\n'.join(lines[start:end + 1])


def phrase_values(block: str) -> set[str]:
    """The template's string values of more than one word: single words are field vocabulary."""
    return {match.group(1) for match in QUOTED.finditer(block)
            if not match.group(2) and ' ' in match.group(1).strip()}


def test_every_hand_template_has_both_controls():
    assert FROZEN_KEYS == [f'frozentemplate-{task}' for task in HAND_TASKS], (
        f'frozen-template arms {FROZEN_KEYS} do not match the hand templates {HAND_TASKS}'
    )
    assert SCHEMA_KEYS == [f'intrinsicmemory-{task}-schema' for task in HAND_TASKS], (
        f'schema-only arms {SCHEMA_KEYS} do not match the hand templates {HAND_TASKS}'
    )


# ── the frozen template ───────────────────────────────────────────────────────

@pytest.mark.parametrize('task', HAND_TASKS)
def test_the_frozen_memory_is_the_hand_templates_json_and_nothing_more(task):
    frozen = build(f'frozentemplate-{task}').template
    hand = hand_prompt(task)

    assert frozen in hand, f'{task}: the frozen memory is not text from the hand template'
    assert frozen.startswith('{') and frozen.endswith('}'), (
        f'{task}: the frozen memory is not one JSON object'
    )
    missing = set(KEY.findall(hand)) - set(KEY.findall(frozen))
    assert not missing, f'{task}: the frozen memory drops template fields {sorted(missing)}'
    for instruction in ('UPDATE INSTRUCTIONS', 'MEMORY UPDATER'):
        assert instruction not in frozen, (
            f'{task}: the frozen memory carries the updater prompt ({instruction!r})'
        )


@pytest.mark.parametrize('key', FROZEN_KEYS)
def test_the_frozen_arm_never_calls_the_model(key):
    llm = FakeLLM(tracker=TokenTracker())
    memory = build(key, llm)

    for _ in range(3):
        memory.summarize(solver_message='sys')

    assert not llm.calls, f'{key} made {len(llm.calls)} LLM calls'


@pytest.mark.parametrize('key', FROZEN_KEYS)
def test_the_frozen_memory_is_the_same_at_every_step_and_every_task(key):
    memory = build(key)
    template = memory.template
    first = memory.summarize(solver_message='sys')
    memory.move_memory_state('go to cabinet 1', 'you arrive at cabinet 1')
    later = memory.summarize(solver_message='sys')
    memory.save_task_context(label=False, feedback='task failed')
    start_task(memory, task='put a pen on the desk')
    next_task = memory.summarize(solver_message='sys')

    for view in (first, later, next_task):
        assert template in view, f'{key}: the agent was not shown the frozen memory'


@pytest.mark.parametrize('task', HAND_TASKS)
def test_the_frozen_memory_sits_where_the_hand_arm_puts_its_memory(task):
    """Only the memory's content may differ between the two arms, never its layout."""
    frozen = build(f'frozentemplate-{task}')
    hand = build(f'intrinsicmemory-{task}', FakeLLM(replies=[frozen.template]))

    assert frozen.summarize(solver_message='sys') == hand.summarize(solver_message='sys')


# ── the schema-only template ──────────────────────────────────────────────────

@pytest.mark.parametrize('task', HAND_TASKS)
def test_the_schema_template_has_the_hand_templates_fields_in_order(task):
    schema = template_block(build(f'intrinsicmemory-{task}-schema').system_prompt)
    hand = template_block(hand_prompt(task))

    assert KEY.findall(schema) == KEY.findall(hand), (
        f'{task}: the schema-only template does not have the hand template\'s fields'
    )


@pytest.mark.parametrize('task', HAND_TASKS)
def test_the_schema_prompt_keeps_none_of_the_hand_templates_values(task):
    schema_prompt = build(f'intrinsicmemory-{task}-schema').system_prompt
    hand = template_block(hand_prompt(task))

    leaked = sorted(value for value in phrase_values(hand) if value in schema_prompt)
    assert not leaked, f'{task}: the schema-only prompt carries hand-written content {leaked}'


@pytest.mark.parametrize('key', SCHEMA_KEYS)
def test_the_schema_template_is_valid_json_with_every_value_empty(key):
    template = json.loads(template_block(build(key).system_prompt))

    def leaves(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from leaves(value)
        elif isinstance(node, list):
            for item in node:
                yield from leaves(item)
        else:
            yield node

    filled = [leaf for leaf in leaves(template) if leaf not in ('', 0, False, None)]
    assert not filled, f'{key}: the schema-only template has values {filled}'


@pytest.mark.parametrize('key', SCHEMA_KEYS)
def test_the_schema_arm_updates_its_memory_through_the_model(key):
    llm = FakeLLM(replies=['an updated memory'], tracker=TokenTracker())
    memory = build(key, llm)

    view = memory.summarize(solver_message='sys')

    assert llm.calls, f'{key} made no memory update'
    assert 'an updated memory' in view, f'{key}: the updated memory did not reach the agent'
