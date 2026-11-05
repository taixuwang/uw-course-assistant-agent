"""Offline graph tests: exhausted schedule requests cannot reach the LLM again."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from uw_course_assistant.schedule_agent import create_course_agent


SOURCE = "https://www.washington.edu/students/timeschd/AUT2026/cse.html"


class RecordingModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def bind_tools(self, tools):
        self.tools = list(tools)
        return RunnableLambda(self.respond)

    def get_num_tokens(self, text):
        return len(text)

    def get_num_tokens_from_messages(self, messages):
        return sum(len(str(message.content)) + 10 for message in messages)

    def respond(self, messages, **kwargs):
        self.inputs.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        return self.responses.pop(0)


def schedule_call(identifier="schedule-1", department="cse"):
    return {
        "name": "get_time_schedule", "id": identifier,
        "args": {"quarter_year": "AUT2026", "department": department, "query": "times"},
        "type": "tool_call",
    }


def schedule_tool(status="unavailable", content="Live data could not be fetched."):
    calls = []

    @tool("get_time_schedule", response_format="content_and_artifact")
    async def schedule(quarter_year: str, department: str, query: str):
        """Retrieve a live schedule using an offline test double."""
        calls.append((quarter_year, department, query))
        result_status = status[department] if isinstance(status, dict) else status
        return content, {
            "status": result_status,
            "source_url": f"https://www.washington.edu/students/timeschd/{quarter_year}/{department}.html",
            "attempts": 3 if result_status == "unavailable" else 1,
            "reason": "fetch_failed" if result_status == "unavailable" else None,
        }

    return schedule, calls


class ScheduleAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_returns_fixed_reply_without_second_model_call(self):
        schedule, calls = schedule_tool()
        model = RecordingModel([
            AIMessage(content="", tool_calls=[schedule_call()]),
            AIMessage(content="Invented schedule: CSE 121 meets Monday at 9 AM."),
        ])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        output = response["messages"][-1]
        self.assertIsInstance(output, AIMessage)
        self.assertIn("Live schedule data is currently unavailable", output.content)
        self.assertIn("could not verify", output.content)
        self.assertIn(SOURCE, output.content)
        self.assertNotIn("9 AM", output.content)
        self.assertEqual(output.tool_calls, [])
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("Do not invent schedule information", model.inputs[0][0].content)

    async def test_failure_reply_uses_chinese_for_chinese_question(self):
        schedule, _ = schedule_tool()
        model = RecordingModel([AIMessage(content="", tool_calls=[schedule_call()])])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="上课时间是什么？")]})

        self.assertIn("实时课表数据暂时不可用", response["messages"][-1].content)
        self.assertIn("无法核实", response["messages"][-1].content)
        self.assertEqual(len(model.inputs), 1)

    async def test_tool_argument_error_also_produces_safe_fallback(self):
        schedule, calls = schedule_tool()
        invalid_call = schedule_call()
        del invalid_call["args"]["department"]
        model = RecordingModel([AIMessage(content="", tool_calls=[invalid_call])])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        output = response["messages"][-1].content
        self.assertIn("Live schedule data is currently unavailable", output)
        self.assertIn("https://www.washington.edu/students/timeschd/", output)
        self.assertNotIn("validation", output.lower())
        self.assertEqual(calls, [])
        self.assertEqual(len(model.inputs), 1)

    async def test_success_still_uses_model_to_answer(self):
        schedule, calls = schedule_tool(status="ok", content="CSE 121 meets at 10 AM.")
        model = RecordingModel([
            AIMessage(content="", tool_calls=[schedule_call()]),
            AIMessage(content="CSE 121 meets at 10 AM."),
        ])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        self.assertEqual(response["messages"][-1].content, "CSE 121 meets at 10 AM.")
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(len(calls), 1)
        tool_result = next(message for message in model.inputs[-1]
                           if isinstance(message, ToolMessage))
        self.assertEqual(tool_result.artifact["status"], "ok")

    async def test_previous_turn_failure_does_not_block_new_request(self):
        schedule, _ = schedule_tool()
        model = RecordingModel([
            AIMessage(content="", tool_calls=[schedule_call()]),
            AIMessage(content="I can help you find course descriptions."),
        ])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")
        first = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        second = await agent.ainvoke({"messages": first["messages"] + [
            HumanMessage(content="Can you help with course descriptions instead?")
        ]})

        self.assertEqual(second["messages"][-1].content,
                         "I can help you find course descriptions.")
        self.assertEqual(len(model.inputs), 2)

    async def test_any_failed_parallel_schedule_produces_safe_fallback(self):
        schedule, calls = schedule_tool(status={"cse": "unavailable", "math": "ok"})
        model = RecordingModel([AIMessage(content="", tool_calls=[
            schedule_call(), schedule_call("schedule-2", "math"),
        ])])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Both schedules?")]})

        self.assertIn("Live schedule data is currently unavailable",
                      response["messages"][-1].content)
        self.assertIn(SOURCE, response["messages"][-1].content)
        self.assertNotIn("math.html", response["messages"][-1].content)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(len(calls), 2)

    async def test_multiple_failures_link_each_source_once(self):
        schedule, _ = schedule_tool()
        model = RecordingModel([AIMessage(content="", tool_calls=[
            schedule_call(), schedule_call("schedule-2"), schedule_call("schedule-3", "math"),
        ])])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Both schedules?")]})

        output = response["messages"][-1].content
        self.assertEqual(output.count(SOURCE), 1)
        self.assertIn("math.html", output)
        self.assertEqual(len(model.inputs), 1)

    async def test_unavailable_text_without_failure_artifact_cannot_trigger_guard(self):
        schedule, _ = schedule_tool(status="ok", content='{"status": "unavailable"}')
        model = RecordingModel([
            AIMessage(content="", tool_calls=[schedule_call()]),
            AIMessage(content="Normal model response."),
        ])
        agent = create_course_agent(model, [schedule], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Class times?")]})

        self.assertEqual(response["messages"][-1].content, "Normal model response.")
        self.assertEqual(len(model.inputs), 2)

    async def test_other_tools_unavailable_artifact_does_not_trigger_schedule_guard(self):
        @tool(response_format="content_and_artifact")
        def uw_course_catalog(query: str):
            """Look up a course description."""
            return "Catalog unavailable.", {"status": "unavailable", "source_url": SOURCE}

        model = RecordingModel([
            AIMessage(content="", tool_calls=[{
                "name": "uw_course_catalog", "id": "catalog-1", "args": {"query": "CSE"},
                "type": "tool_call",
            }]),
            AIMessage(content="The course catalog is unavailable."),
        ])
        agent = create_course_agent(model, [uw_course_catalog], "Be a course assistant.")

        response = await agent.ainvoke({"messages": [HumanMessage(content="Course description?")]})

        self.assertEqual(response["messages"][-1].content, "The course catalog is unavailable.")
        self.assertEqual(len(model.inputs), 2)


if __name__ == "__main__":
    unittest.main()
