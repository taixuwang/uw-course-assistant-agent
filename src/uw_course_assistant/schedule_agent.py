"""Agent construction with a deterministic response to unavailable live schedules."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent


SCHEDULE_FAILURE_INSTRUCTIONS = (
    "Only report quarter-specific class times, locations, instructors, and seat "
    "availability supported by a successful get_time_schedule result for that "
    "quarter and department. If get_time_schedule reports status unavailable, "
    "explicitly tell the user that real-time schedule data is unavailable and "
    "that these details could not be verified. Do not invent schedule information "
    "or substitute static catalog data, model knowledge, or earlier conversation "
    "as current schedule evidence. Unavailable data does not mean no courses or "
    "no seats exist. The tool already performs bounded retries; do not call it "
    "again for the same failed request in this turn. Offer the official source "
    "link or suggest trying again later."
)


def _schedule_unavailable_reply(messages):
    """Use tool artifacts/status from this user turn, never scraped error text."""
    current_turn = []
    user_text = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_text = str(message.content)
            break
        current_turn.append(message)

    failures = []
    for message in reversed(current_turn):
        if not isinstance(message, ToolMessage) or message.name != "get_time_schedule":
            continue
        if isinstance(message.artifact, dict) and message.artifact.get("status") == "unavailable":
            failures.append(message.artifact)
        elif message.status == "error":
            # Invalid tool arguments fail before the coroutine can make an artifact.
            failures.append({"source_url": "https://www.washington.edu/students/timeschd/"})
    if not failures:
        return None

    urls = list(dict.fromkeys(
        failure["source_url"]
        for failure in failures
        if isinstance(failure.get("source_url"), str) and failure["source_url"]
    ))
    if any("\u4e00" <= character <= "\u9fff" for character in user_text):
        reply = (
            "请求的实时课表数据暂时不可用，我无法核实上课时间、地点、教师或余位。"
            "这不代表没有课程或没有余位。请稍后重试，或直接查看 UW 官方课表。"
        )
    else:
        reply = (
            "Live schedule data is currently unavailable for the requested schedule "
            "page(s). I could not verify class times, locations, instructors, or seat "
            "availability. This does not mean there are no courses or open seats. "
            "Please try again later or check the official UW Time Schedule directly."
        )
    if urls:
        reply += "\n\n" + "\n".join(f"- [UW Time Schedule]({url})" for url in urls)
    return reply


def create_course_agent(llm, tools, prompt):
    """End the turn on any failed schedule, including mixed parallel tool calls."""
    bound_model = llm.bind_tools(tools)

    def select_model(state, runtime):
        failure_reply = _schedule_unavailable_reply(state["messages"])
        if failure_reply is not None:
            return RunnableLambda(lambda _messages: AIMessage(content=failure_reply))
        return bound_model

    return create_react_agent(
        select_model, tools, prompt=f"{prompt}\n{SCHEDULE_FAILURE_INSTRUCTIONS}"
    )
