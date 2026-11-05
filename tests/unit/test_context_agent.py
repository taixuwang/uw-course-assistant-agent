"""Offline graph regressions for token budgeting around tool execution."""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from uw_course_assistant.context import ContextBudget
from uw_course_assistant.schedule_agent import create_course_agent


class BudgetRecordingModel:
    """Use character counts as deterministic tokens without APIs or downloads."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []
        self.output_limits = []
        self.summary_inputs = []
        self.summary_limits = []

    def get_num_tokens(self, text):
        return len(text)

    def get_num_tokens_from_messages(self, messages, **kwargs):
        return sum(
            len(str(message.content)) + 10
            + (len(json.dumps(message.tool_calls)) if isinstance(message, AIMessage)
               and message.tool_calls else 0)
            for message in messages
        )

    def bind_tools(self, tools, **kwargs):
        self.tools = list(tools)
        return RunnableLambda(self.respond)

    def bind(self, **kwargs):
        def summarize(messages):
            self.summary_inputs.append(list(messages))
            self.summary_limits.append(kwargs.get("max_tokens"))
            return AIMessage(content="The user wants CSE courses without prerequisites.")

        return RunnableLambda(summarize)

    def respond(self, messages, **kwargs):
        self.inputs.append(list(messages))
        self.output_limits.append(kwargs.get("max_tokens"))
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        return self.responses.pop(0)


def catalog_call(identifier="catalog-1"):
    return {
        "name": "uw_course_catalog", "id": identifier,
        "args": {"query": "CSE courses without prerequisites"}, "type": "tool_call",
    }


def old_turn():
    return [
        HumanMessage(content="No prerequisites. " + "u" * 1800),
        AIMessage(content="Earlier catalog advice. " + "a" * 1200),
    ]


def make_budget():
    return ContextBudget(
        context_window_tokens=6000,
        max_output_tokens=200,
        safety_margin_tokens=50,
        summary_tokens=200,
    )


class ContextAgentTests(unittest.IsolatedAsyncioTestCase):
    def assert_requests_within_budget(self, model, budget):
        self.assertEqual(model.output_limits, [budget.max_output_tokens] * len(model.inputs))
        schema_tokens = model.get_num_tokens(json.dumps(
            [convert_to_openai_tool(item) for item in model.tools], ensure_ascii=False,
        ))
        for messages in model.inputs:
            self.assertLessEqual(
                model.get_num_tokens_from_messages(messages) + schema_tokens
                + budget.max_output_tokens + budget.safety_margin_tokens,
                budget.context_window_tokens,
            )
        for messages in model.summary_inputs:
            self.assertLessEqual(
                model.get_num_tokens_from_messages(messages) + budget.summary_tokens
                + budget.safety_margin_tokens,
                budget.context_window_tokens,
            )

    def assert_paired_tools(self, messages):
        pending = set()
        for message in messages:
            if isinstance(message, ToolMessage):
                self.assertIn(message.tool_call_id, pending)
                pending.remove(message.tool_call_id)
            else:
                self.assertFalse(pending, "A tool call was separated from its result")
                if isinstance(message, AIMessage):
                    pending.update(call["id"] for call in message.tool_calls)
        self.assertFalse(pending)

    async def test_tool_results_trigger_compaction_before_next_model_call(self):
        @tool
        def uw_course_catalog(query: str):
            """Look up static course details."""
            return "CSE 121 has no prerequisites. " + "t" * 2000

        model = BudgetRecordingModel([
            AIMessage(content="", tool_calls=[catalog_call()]),
            AIMessage(content="CSE 121 is suitable."),
            AIMessage(content="You asked for courses without prerequisites."),
        ])
        budget = make_budget()
        agent = create_course_agent(
            model, [uw_course_catalog], "Help with UW courses.", context_budget=budget,
        )

        first = await agent.ainvoke({"messages": old_turn() + [
            HumanMessage(content="Find CSE courses without prerequisites."),
        ]})

        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(model.summary_inputs, "Tools must not bypass the token budget")
        self.assertTrue(any("u" * 100 in str(m.content) for m in model.inputs[0]))
        self.assertFalse(any("u" * 100 in str(m.content) for m in model.inputs[1]))
        self.assertTrue(any(isinstance(m, ToolMessage) for m in model.inputs[1]))
        self.assert_paired_tools(model.inputs[1])
        self.assertFalse(any("u" * 100 in str(m.content) for m in first["messages"]))
        self.assertEqual(first["messages"][-1].content, "CSE 121 is suitable.")
        self.assertTrue(all(limit == budget.summary_tokens for limit in model.summary_limits))

        await agent.ainvoke({"messages": first["messages"] + [
            HumanMessage(content="What constraint did I give you?"),
        ]})

        self.assertFalse(any("u" * 100 in str(m.content) for m in model.inputs[-1]))
        self.assertEqual(sum("The user wants CSE courses without prerequisites." in str(m.content)
                             for m in model.inputs[-1]), 1)
        self.assert_paired_tools(model.inputs[-1])
        self.assert_requests_within_budget(model, budget)

    async def test_parallel_tool_calls_and_results_survive_as_one_group(self):
        @tool
        def uw_course_catalog(query: str):
            """Look up static course details."""
            return "A matching course. " + "t" * 1000

        model = BudgetRecordingModel([
            AIMessage(content="", tool_calls=[catalog_call(), catalog_call("catalog-2")]),
            AIMessage(content="Here are the matching courses."),
        ])
        agent = create_course_agent(
            model, [uw_course_catalog], "Help with UW courses.", context_budget=make_budget(),
        )

        response = await agent.ainvoke({"messages": old_turn() + [
            HumanMessage(content="Search both catalogs."),
        ]})

        self.assertTrue(model.summary_inputs)
        self.assert_paired_tools(model.inputs[-1])
        self.assertEqual(
            {m.tool_call_id for m in model.inputs[-1] if isinstance(m, ToolMessage)},
            {"catalog-1", "catalog-2"},
        )
        self.assert_paired_tools(response["messages"])
        self.assert_requests_within_budget(model, make_budget())

    async def test_current_turn_over_budget_stops_before_any_model_request(self):
        model = BudgetRecordingModel([AIMessage(content="Must not be requested.")])
        agent = create_course_agent(
            model, [], "Help with UW courses.", context_budget=make_budget(),
        )

        with self.assertRaises(ValueError):
            await agent.ainvoke({"messages": [HumanMessage(content="x" * 8000)]})

        self.assertEqual(model.inputs, [])
        self.assertEqual(model.summary_inputs, [])

    async def test_oversized_current_tool_result_stops_before_next_model_request(self):
        @tool
        def uw_course_catalog(query: str):
            """Look up static course details."""
            return "t" * 8000

        model = BudgetRecordingModel([
            AIMessage(content="", tool_calls=[catalog_call()]),
            AIMessage(content="Must not be requested."),
        ])
        agent = create_course_agent(
            model, [uw_course_catalog], "Help with UW courses.", context_budget=make_budget(),
        )

        with self.assertRaises(ValueError):
            await agent.ainvoke({"messages": [HumanMessage(content="Find CSE courses.")]})

        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.summary_inputs, [])

    async def test_oversized_failed_schedule_keeps_deterministic_failure_guard(self):
        @tool("get_time_schedule", response_format="content_and_artifact")
        async def schedule(query: str):
            """Look up the live schedule."""
            return "Failure details. " + "e" * 8000, {
                "status": "unavailable",
                "source_url": "https://www.washington.edu/students/timeschd/",
            }

        model = BudgetRecordingModel([AIMessage(content="", tool_calls=[{
            "name": "get_time_schedule", "id": "schedule-1", "args": {"query": "CSE"},
            "type": "tool_call",
        }])])
        agent = create_course_agent(
            model, [schedule], "Help with UW courses.", context_budget=make_budget(),
        )

        response = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        self.assertIn("Live schedule data is currently unavailable", response["messages"][-1].content)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.summary_inputs, [])
        self.assert_paired_tools(response["messages"])


if __name__ == "__main__":
    unittest.main()
