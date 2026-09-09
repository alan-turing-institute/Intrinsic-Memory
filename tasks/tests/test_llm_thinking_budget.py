"""A reasoning model has to be stopped thinking, and its trace has to be found.

Two independent failures leave a reasoning model unable to answer at all, and
both are invisible from the result files: every reply comes back with no content,
and an experiment still writes its row.

The trace arrives under a field name that differs by server. vLLM 0.15 calls it
`reasoning_content`, vLLM 0.28 calls it `reasoning`; whichever it is, a reply
carrying one and no content is a budget that went entirely on reasoning, and the
retry that grows the budget turns on recognising that.

Stopping the thinking is the endpoint's job, through the per-request
`thinking_token_budget` vLLM takes alongside a `--reasoning-config`. Servers that
do not implement it refuse the field, so it is dropped and remembered the way
temperature and the stop sequence are.
"""

import pytest

from mas.llm import Message

from tasks.tests.fakes import (
    FakeCompletions,
    StarvedReasoningCompletions,
    chat_over_fake_completions,
)
from tasks.tests.test_llm_layer import budgets, call_settings

PROMPT = [Message("user", "what is the next action?")]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_a_trace_under_either_field_name_is_recognised_as_a_starved_budget(field):
    """The retry has to grow the budget whichever name the trace arrives under."""
    starved = StarvedReasoningCompletions([" go to desk 1"], needs=2048, reasoning_field=field)
    chat, completions = chat_over_fake_completions(
        None, completions=starved,
        settings=call_settings(max_tokens=512, max_tokens_ceiling=8192),
    )

    answer = chat(PROMPT)

    assert answer.strip() == "go to desk 1", (
        f"a trace under `{field}` left the call unanswered; the budget never grew"
    )
    assert budgets(completions) == [512, 1024, 2048], (
        f"budgets asked for were {budgets(completions)}, not a climb from 512"
    )


def test_a_reply_with_neither_content_nor_a_trace_is_not_blamed_on_the_budget():
    """Nothing to grow towards, so growing the budget only repeats the bill."""
    chat, completions = chat_over_fake_completions(
        [None], settings=call_settings(max_tokens=512, max_tokens_ceiling=8192),
    )

    with pytest.raises(Exception):
        chat(PROMPT)

    assert budgets(completions) == [512] * len(completions.calls), (
        f"budgets asked for were {budgets(completions)}; a reply with no trace says "
        f"nothing about the budget being too small"
    )


def test_a_configured_thinking_budget_is_sent_with_every_call():
    chat, completions = chat_over_fake_completions(
        ["go to desk 1"], settings=call_settings(thinking_token_budget=1024),
    )

    chat(PROMPT)
    chat(PROMPT)

    assert [call.get("extra_body") for call in completions.calls] == [
        {"thinking_token_budget": 1024}
    ] * 2, f"the thinking budget did not reach the endpoint: {completions.calls}"


def test_no_thinking_budget_configured_sends_no_such_field():
    """A model that does not reason must see the request it saw before."""
    chat, completions = chat_over_fake_completions(["go to desk 1"])

    chat(PROMPT)

    assert "extra_body" not in completions.calls[0], (
        "an unconfigured thinking budget still changed the request"
    )


class ThinkingBudgetRejectingCompletions(FakeCompletions):
    """An endpoint that refuses the `thinking_token_budget` field.

    The shape a vLLM without a `--reasoning-config` sends: a 400 naming the
    field. Every attempt is appended to `calls`, refused ones included.
    """

    def __init__(self, script):
        super().__init__(script)
        self.refusals = 0

    def create(self, **kwargs):
        if kwargs.get("extra_body", {}).get("thinking_token_budget"):
            self.calls.append(kwargs)
            self.refusals += 1
            error = ValueError(
                "1 validation error for ChatCompletionRequest\\nthinking_token_budget\\n  "
                "Extra inputs are not permitted"
            )
            error.status_code = 400
            raise error
        return super().create(**kwargs)


def test_an_endpoint_that_refuses_the_thinking_budget_is_answered_without_it():
    refusing = ThinkingBudgetRejectingCompletions(["go to desk 1"])
    chat, _ = chat_over_fake_completions(
        None, completions=refusing, settings=call_settings(thinking_token_budget=1024),
    )

    assert chat(PROMPT) == "go to desk 1", "a refused field cost the call its answer"


def test_the_refusal_costs_one_request_not_one_per_call():
    refusing = ThinkingBudgetRejectingCompletions(["go to desk 1"])
    chat, _ = chat_over_fake_completions(
        None, completions=refusing, settings=call_settings(thinking_token_budget=1024),
    )

    for _ in range(4):
        chat(PROMPT)

    assert refusing.refusals == 1, (
        f"the thinking budget was sent {refusing.refusals} times; a refusal should be "
        f"remembered for the life of the client"
    )
