"""The LLM-generated memory template reaches the memory-update prompt.

`intrinsicmemory-llm-structured-template` asks the model to write a memory
template for the task, then runs every later update against it. The other
intrinsic modules carry a fixed template or none, and must not receive one.
"""

import tempfile

import pytest

from mas.memory import MASMemoryBase
from mas.memory.mas_memory.intrinsicmemory_llm_structured_template import (
    IntrinsicMASMemoryLLMTemplate,
)
from mas.memory.mas_memory.intrinsicmemory import IntrinsicMASMemoryNoTemplate
from mas.memory.mas_memory.prompt import INTRINSICMEMORY_LLM_TEMPLATE

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM

TEMPLATE = "## Location\n{}\n## Items held\n{}"
TRAJECTORY = "\n\n>go to desk 1\nOn the desk you see a mug.\n>"


def build_memory(memory_cls=IntrinsicMASMemoryLLMTemplate, replies=None):
    llm = FakeLLM(replies=replies or [TEMPLATE, "updated memory"])
    memory = memory_cls(
        namespace="template-test",
        global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=llm,
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context("put the mug on the desk", "a description of the task")
    memory.current_task_context.task_trajectory = TRAJECTORY
    return memory, llm


def prompts_sent(llm) -> list[str]:
    """The user-role content of every call, in order."""
    return [
        "\n".join(message.content for message in call if message.role == "user")
        for call in llm.calls
    ]


# ── the template is generated ─────────────────────────────────────────────────

def test_the_first_summarize_asks_the_model_for_a_template():
    memory, llm = build_memory()

    memory.summarize()

    assert llm.calls, "no LLM call was made"
    creation = INTRINSICMEMORY_LLM_TEMPLATE.template_creation_prompt.split("{")[0].strip()
    assert creation[:40] in prompts_sent(llm)[0], (
        "the first call was not the template-creation call"
    )


def test_the_template_is_generated_once_per_task():
    memory, llm = build_memory(replies=[TEMPLATE, "a", "b", "c", "d"])

    for _ in range(4):
        memory.summarize()

    creation = INTRINSICMEMORY_LLM_TEMPLATE.template_creation_prompt.split("{")[0].strip()[:40]
    generations = [prompt for prompt in prompts_sent(llm) if creation in prompt]
    assert len(generations) == 1, (
        f"the template was regenerated {len(generations)} times in one task"
    )


# ── A5 · and it reaches the update prompt ─────────────────────────────────────

def test_the_generated_template_is_kept_where_the_update_reads_it():
    memory, _ = build_memory()

    memory.summarize()

    assert memory.memory_template == TEMPLATE, (
        f"memory_template holds {memory.memory_template!r}; the update prompt reads this field"
    )


def test_the_update_prompt_is_the_module_prompt_with_the_template_filled_in():
    """Exact render: the template also appears if it has merely displaced the
    accumulated memory, so only its position distinguishes the two."""
    memory, llm = build_memory()

    memory.summarize()

    updates = prompts_sent(llm)[1:]
    assert updates, "no memory-update call followed the template generation"
    expected = memory.memory_update_prompt.format(
        custom_message="",
        template_instructions=INTRINSICMEMORY_LLM_TEMPLATE.memory_template_section.format(
            template_instructions=TEMPLATE
        ),
        task_description="a description of the task",
        task_trajectory=TRAJECTORY,
        current_memory="",
        word_limit="",
    )
    assert expected.strip() in updates[0], (
        "the first update prompt is not this module's prompt carrying the "
        "generated template and an empty memory"
    )


def test_the_update_prompt_has_somewhere_to_put_the_template():
    """A field holding the template is not enough if the prompt has no slot."""
    memory, _ = build_memory()

    assert "{template_instructions}" in memory.memory_update_prompt, (
        "the module's update prompt has no template_instructions slot, so the "
        "template is formatted into nothing"
    )


def test_the_accumulated_memory_is_what_the_update_call_returned():
    memory, _ = build_memory()

    memory.summarize()

    assert memory.agent_intrinsic_memory == "updated memory"


# ── the other modules are unaffected ──────────────────────────────────────────

@pytest.mark.parametrize("memory_cls", [MASMemoryBase, IntrinsicMASMemoryNoTemplate])
def test_a_module_with_no_template_does_not_acquire_one(memory_cls):
    memory, llm = build_memory(memory_cls=memory_cls, replies=["updated memory"])

    memory.summarize()

    assert not any(TEMPLATE in prompt for prompt in prompts_sent(llm))
