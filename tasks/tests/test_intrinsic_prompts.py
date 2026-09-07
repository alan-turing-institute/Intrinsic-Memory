"""Each registered intrinsic memory module carries the prompts it is named for.

The modules in this family differ from one another only by which prompt
constants they hold, and a wrong one raises nothing, logs nothing and passes the
compatibility matrix - it just runs a whole experiment against the wrong memory
instructions.

Both properties are asserted per registry key, so a module added to
`MAS_MEMORY_MODULES` without an entry here fails rather than going uncovered.
"""

import tempfile

import pytest

from mas.module_map import MAS_MEMORY_MODULES
from mas.memory.mas_memory.prompt import (
    INTRINSICMEMORY_ALFWORLD,
    INTRINSICMEMORY_BABYAI,
    INTRINSICMEMORY_DEFAULT,
    INTRINSICMEMORY_FEVER,
    INTRINSICMEMORY_HOTPOTQA,
    INTRINSICMEMORY_JERICHO,
    INTRINSICMEMORY_LLM_TEMPLATE,
    INTRINSICMEMORY_NOTEMPLATE,
    INTRINSICMEMORY_PDDL,
    INTRINSICMEMORY_SCIWORLD,
    INTRINSICMEMORY_SWEBENCH,
)

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM

SYSTEM_PROMPT_OWNER = {
    'intrinsicmemory-pddl': INTRINSICMEMORY_PDDL,
    'intrinsicmemory-fever': INTRINSICMEMORY_FEVER,
    'intrinsicmemory-hotpotqa': INTRINSICMEMORY_HOTPOTQA,
    'intrinsicmemory-alfworld': INTRINSICMEMORY_ALFWORLD,
    'intrinsicmemory-babyai': INTRINSICMEMORY_BABYAI,
    'intrinsicmemory-jericho': INTRINSICMEMORY_JERICHO,
    'intrinsicmemory-sciworld': INTRINSICMEMORY_SCIWORLD,
    'intrinsicmemory-swebench': INTRINSICMEMORY_SWEBENCH,
    'intrinsicmemory-notemplate': INTRINSICMEMORY_NOTEMPLATE,
    'intrinsicmemory-llm-structured-template': INTRINSICMEMORY_LLM_TEMPLATE,
}

# One shared update prompt across the family: the memory template is the variable
# under test, so the update procedure is held constant.
UPDATE_PROMPT_OWNER = dict.fromkeys(SYSTEM_PROMPT_OWNER, INTRINSICMEMORY_DEFAULT)

INTRINSIC_KEYS = sorted(key for key in MAS_MEMORY_MODULES if key.startswith('intrinsicmemory'))


def build(key: str):
    return MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
        llm_model=FakeLLM(),
        embedding_func=FakeEmbeddingFunc(),
    )


def test_every_registered_intrinsic_module_has_an_expected_prompt():
    """A module added to the registry must be named here rather than go uncovered."""
    missing = set(INTRINSIC_KEYS) - set(SYSTEM_PROMPT_OWNER)
    assert not missing, f"registered but no expected system prompt: {sorted(missing)}"

    stale = set(SYSTEM_PROMPT_OWNER) - set(INTRINSIC_KEYS)
    assert not stale, f"expected a system prompt for unregistered modules: {sorted(stale)}"

    assert set(UPDATE_PROMPT_OWNER) == set(SYSTEM_PROMPT_OWNER), (
        "every module needs an expected update prompt as well as a system prompt"
    )


# ── A1 · the system prompt ────────────────────────────────────────────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_module_carries_its_own_system_prompt(key):
    expected = SYSTEM_PROMPT_OWNER[key].system_prompt

    assert build(key).system_prompt == expected, (
        f"{key} does not carry the system prompt recorded for it"
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_system_prompt_is_not_empty(key):
    """The base leaves it empty, so an unset override is indistinguishable from none."""
    assert build(key).system_prompt, f"{key} has an empty system_prompt"


# Neither of these carries a fixed template, so they share the generic summariser
# prompt and differ by behaviour instead. Every other intrinsic module is named
# for a dataset and must have a template of its own.
TEMPLATELESS = ('intrinsicmemory-notemplate', 'intrinsicmemory-llm-structured-template')

TASK_SPECIFIC_KEYS = [key for key in INTRINSIC_KEYS if key not in TEMPLATELESS]


def test_the_task_specific_modules_each_have_their_own_system_prompt():
    """The template is what these differ by, so no two may share one.

    Whether a hand-written template beats the one an LLM writes is the experiment,
    so two datasets sharing a template would compare the wrong thing while looking
    like a full result. Derived from the registry rather than listed, so a new
    dataset's module is covered by being registered.
    """
    prompts = {key: build(key).system_prompt for key in TASK_SPECIFIC_KEYS}

    collisions = [
        (a, b) for a in prompts for b in prompts if a < b and prompts[a] == prompts[b]
    ]
    assert not collisions, f"modules sharing one system prompt: {collisions}"


# ── A1b · the update prompt ───────────────────────────────────────────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_module_carries_the_expected_update_prompt(key):
    expected = UPDATE_PROMPT_OWNER[key].memory_update_prompt

    assert build(key).memory_update_prompt == expected, (
        f"{key} does not carry the update prompt recorded for it"
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_update_prompt_accepts_every_field_summarize_formats(key):
    """`summarize` formats five fields in; a prompt missing a slot drops one silently."""
    prompt = build(key).memory_update_prompt

    formatted = prompt.format(
        custom_message='CUSTOM',
        template_instructions='TEMPLATE',
        task_description='DESCRIPTION',
        task_trajectory='TRAJECTORY',
        current_memory='MEMORY',
    )

    for field in ('DESCRIPTION', 'TRAJECTORY', 'MEMORY'):
        assert field in formatted, f"{key}'s update prompt drops {field}"

# ── a memory built from another's class carries the same prompts ──────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_a_rebuilt_memory_keeps_its_prompts(key):
    """A module's prompts belong to its class, not to one instance of it.

    So a second instance built from the first one's class carries them too. This
    is what `build_system` relies on to give the validator its own memory.
    """
    original = build(key)

    rebuilt = original.__class__(
        namespace=original.namespace + '_validator',
        global_config=original.global_config,
        llm_model=original.llm_model,
        embedding_func=original.embedding_func,
    )

    assert rebuilt.system_prompt == original.system_prompt, (
        f"{key} lost its system prompt when rebuilt from its own class"
    )
    assert rebuilt.memory_update_prompt == original.memory_update_prompt, (
        f"{key} lost its update prompt when rebuilt from its own class"
    )
