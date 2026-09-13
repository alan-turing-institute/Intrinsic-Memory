"""What a finished task costs to keep.

Every memory module that stores anything stores `MASMessage.to_dict`, and it is
kept for the rest of the experiment. What goes in it therefore has to be what a
later task reads back.
"""

from unittest.mock import MagicMock, patch

import pytest

from mas.memory.common import MASMessage
from mas.memory.mas_memory.intrinsicmemory import IntrinsicMASMemoryNoTemplate
from mas.reasoning import ReasoningBase

from tasks.mas_workflow.autogen.autogen import AutoGen

PATCH_GPTCHAT = "mas.memory.mas_memory.intrinsicmemory.GPTChat"


class StubReasoning(ReasoningBase):

    def __init__(self, response: str = "go north"):
        super().__init__(llm_model=MagicMock())
        self._response = response

    def __call__(self, prompts, config):
        return self._response


def run_episode(tmp_path, few_shots: list[str], trials: int = 4) -> MASMessage:
    """One episode, returning the task context its memory module would store."""
    llm = MagicMock()
    llm.return_value = "updated memory"
    llm.model_name = "test-model"
    memory = IntrinsicMASMemoryNoTemplate(
        namespace="test",
        global_config={"working_dir": str(tmp_path)},
        llm_model=llm,
        embedding_func=MagicMock(),
    )

    env = MagicMock()
    env.max_trials = trials
    env.process_action.side_effect = lambda action: action.strip()
    env.is_thought.return_value = False
    env.step.return_value = ("You are in a field.", 0.0, False)
    env.feedback.return_value = (0.0, False, "you did not finish")

    workflow = AutoGen()
    workflow.build_system(
        StubReasoning(),
        memory,
        env,
        config={"successful_topk": 1, "failed_topk": 0, "insights_topk": 3,
                "threshold": 0, "use_projector": False, "use_validator": False},
    )

    with patch(PATCH_GPTCHAT):
        workflow.schedule(
            {"task_main": "t", "task_description": "d", "few_shots": few_shots}
        )

    return memory.current_task_context


@pytest.fixture
def stored(tmp_path):
    def store(few_shots: list[str], trials: int = 4) -> dict:
        return MASMessage.to_dict(run_episode(tmp_path, few_shots, trials))
    return store


def test_stored_task_carries_no_agent_prompt(stored):
    """Nothing a task stores repeats the prompt the agent was given."""
    marker = "ONLY-IN-THE-PROMPT"

    stored_task = stored([f"an example mentioning {marker}"])

    assert marker not in stored_task["state_chain"], (
        "the agent's prompt is in the stored interaction graph; it is written once "
        "per trial, so a stored task grows with the square of the trial budget"
    )


def test_stored_task_size_is_independent_of_prompt_size(stored):
    """A longer prompt costs nothing to store: only the trajectory is kept."""
    short = len(stored(["x"])["state_chain"])
    long = len(stored(["x" * 50_000])["state_chain"])

    assert long == short, (
        f"storing a task grew by {long - short} bytes when its prompt grew by "
        f"50,000, so prompt length is what a stored experiment costs"
    )


def test_stored_task_carries_its_trajectory(stored):
    """The actions and observations a later task reads back are still stored."""
    stored_task = stored(["an example"])

    assert "go north" in stored_task["state_chain"]
    assert "You are in a field." in stored_task["state_chain"]
