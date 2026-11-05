"""Offline budget and message-integrity checks with deterministic token counts."""

import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from uw_course_assistant.context import (
    SUMMARY_HEADER, SUMMARY_ID, ContextBudget, ContextBudgetExceeded, TokenBudgetContext,
)


class CountingModel:
    def __init__(self, summary="User prefers courses without prerequisites."):
        self.summary = summary
        self.requests = []

    def get_num_tokens(self, text):
        return len(text)

    def get_num_tokens_from_messages(self, messages):
        return 3 + sum(
            len(str(m.content)) + 8
            + (len(json.dumps(m.tool_calls)) if isinstance(m, AIMessage) and m.tool_calls else 0)
            for m in messages
        )

    def bind(self, **kwargs):
        def summarize(messages):
            self.requests.append((list(messages), kwargs))
            return AIMessage(content=self.summary)
        return RunnableLambda(summarize)


def make_context(model=None, tools=(), budget=None):
    return TokenBudgetContext(model or CountingModel(), tools, "Help choose UW courses.", budget or ContextBudget(
        context_window_tokens=6000, max_output_tokens=200,
        safety_margin_tokens=50, summary_tokens=200,
    ))


def run(coroutine):
    # The project does not require pytest-asyncio.
    import asyncio
    return asyncio.run(coroutine)


def call(identifier):
    return {"name": "catalog", "args": {"query": "CSE"}, "id": identifier, "type": "tool_call"}


def test_budget_counts_prompt_schemas_and_output_reserve_at_exact_boundary():
    @tool
    def catalog(query: str):
        """Search catalog descriptions and prerequisites."""
        return query

    model = CountingModel()
    context = make_context(model, [catalog])
    messages = [HumanMessage(content="CSE")]
    count = context.input_token_count(messages)
    assert count > model.get_num_tokens_from_messages(messages)
    assert context.tool_tokens > 0
    context.budget = replace(context.budget, context_window_tokens=count + 200 + 50)
    assert run(context.compact(messages)) == messages
    assert model.requests == []
    context.budget = replace(context.budget, context_window_tokens=count + 200 + 49)
    with pytest.raises(ContextBudgetExceeded):
        run(context.compact(messages))
    assert model.requests == []


def test_many_short_messages_do_not_trigger_message_count_compression():
    model = CountingModel()
    context = make_context(model)
    messages = [m for _ in range(80) for m in [HumanMessage(content="Hi"), AIMessage(content="Hi")]]
    assert run(context.compact(messages)) == messages
    assert model.requests == []


def test_few_long_messages_trigger_compression_and_pin_current_turn():
    model = CountingModel()
    context = make_context(model)
    current = HumanMessage(content="Find no-prerequisite courses. " + "q" * 1800)
    messages = [HumanMessage(content="old" + "u" * 2000), AIMessage(content="a" * 2000), current]
    result = run(context.compact(messages))
    assert result[-1] is current
    assert result[0].id == SUMMARY_ID
    assert len(result) == 2
    assert context.input_token_count(result) <= context.budget.input_tokens
    assert len(model.requests) == 1


def test_old_parallel_and_sequential_tool_exchanges_are_summarized_together():
    model = CountingModel()
    context = make_context(model)
    old = [
        HumanMessage(content="Earlier search."),
        AIMessage(content="", tool_calls=[call("a"), call("b")]),
        ToolMessage(content="result b " + "b" * 1500, tool_call_id="b"),
        ToolMessage(content="result a " + "a" * 1500, tool_call_id="a"),
        AIMessage(content="", tool_calls=[call("c")]),
        ToolMessage(content="result c", tool_call_id="c"),
        AIMessage(content="Earlier answer."),
    ]
    current = HumanMessage(content="new request " + "n" * 2700)
    result = run(context.compact([*old, current]))
    assert result[-1] is current
    assert not any(isinstance(m, ToolMessage) for m in result)
    request, _ = model.requests[0]
    assert all(not isinstance(m, ToolMessage) and not getattr(m, "tool_calls", []) for m in request)
    payload = json.loads(request[-1].content)
    records = payload["older_turns"][0]
    assert [r["tool_call_id"] for r in records if r["role"] == "tool"] == ["b", "a", "c"]
    assert "result a" in request[-1].content and "result b" in request[-1].content


def test_pinned_instructions_and_newest_complete_turn_survive():
    context = make_context()
    pinned = SystemMessage(content="Preserve these instructions.")
    old = [HumanMessage(content="u" * 1900), AIMessage(content="a" * 2000)]
    recent = [HumanMessage(content="Recent question."), AIMessage(content="Recent answer.")]
    current = HumanMessage(content="q" * 2000)
    result = run(context.compact([pinned, *old, *recent, current]))
    assert result[0] is pinned
    assert result[-3:] == [*recent, current]
    assert len([m for m in result if m.id == SUMMARY_ID]) == 1


def test_previous_summary_is_updated_instead_of_accumulated():
    model = CountingModel()
    context = make_context(model)
    previous = SystemMessage(id=SUMMARY_ID, content=SUMMARY_HEADER + "User avoids mornings.")
    messages = [previous, HumanMessage(content="u" * 2000), AIMessage(content="a" * 2000), HumanMessage(content="q" * 1800)]
    result = run(context.compact(messages))
    assert "User avoids mornings." in model.requests[0][0][-1].content
    assert sum(m.id == SUMMARY_ID for m in result) == 1
    assert previous not in result


def test_summary_requests_are_batched_with_complete_turns_and_within_budget():
    model = CountingModel()
    context = make_context(model)
    messages = []
    for index in range(8):
        messages.extend([HumanMessage(content=f"turn {index} " + "u" * 900), AIMessage(content="a" * 1000)])
    messages.append(HumanMessage(content="Now find courses."))
    result = run(context.compact(messages))
    assert len(model.requests) > 1
    for request, kwargs in model.requests:
        assert model.get_num_tokens_from_messages(request) + kwargs["max_tokens"] + context.budget.safety_margin_tokens <= context.budget.context_window_tokens
        for records in json.loads(request[-1].content)["older_turns"]:
            assert [r["role"] for r in records] == ["human", "ai"]
    assert context.input_token_count(result) <= context.budget.input_tokens


def test_single_oversized_old_turn_is_rejected_before_summarization():
    model = CountingModel()
    context = make_context(model)
    messages = [HumanMessage(content="x" * 6000), AIMessage(content="Old answer"), HumanMessage(content="New question")]
    with pytest.raises(ContextBudgetExceeded, match="older turn"):
        run(context.compact(messages))
    assert model.requests == []


@pytest.mark.parametrize("summary", ["", "s" * 201])
def test_unusable_or_oversized_summary_cannot_replace_original_history(summary):
    model = CountingModel(summary)
    context = make_context(model)
    messages = [HumanMessage(content="u" * 2000), AIMessage(content="a" * 2000), HumanMessage(content="q" * 1800)]
    original = list(messages)
    with pytest.raises(ContextBudgetExceeded):
        run(context.compact(messages))
    assert messages == original


@pytest.mark.parametrize("messages", [
    [HumanMessage(content="Hi"), AIMessage(content="", tool_calls=[call("a")])],
    [HumanMessage(content="Hi"), ToolMessage(content="orphan", tool_call_id="a")],
    [HumanMessage(content="Hi"), AIMessage(content="", tool_calls=[call("a"), call("a")])],
    [HumanMessage(content="Hi"), AIMessage(content="", tool_calls=[call("a")]), HumanMessage(content="Next")],
    [HumanMessage(content="Hi"), AIMessage(content="", tool_calls=[call("a")]), ToolMessage(content="wrong", tool_call_id="b")],
])
def test_invalid_tool_pairs_are_rejected_even_under_budget(messages):
    model = CountingModel()
    with pytest.raises(ValueError, match="[Tt]ool"):
        run(make_context(model).compact(messages))
    assert model.requests == []


@pytest.mark.parametrize("kwargs", [
    {"context_window_tokens": 0}, {"max_output_tokens": -1}, {"summary_tokens": True},
    {"safety_margin_tokens": -1}, {"context_window_tokens": 100},
])
def test_invalid_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ContextBudget(**kwargs)
