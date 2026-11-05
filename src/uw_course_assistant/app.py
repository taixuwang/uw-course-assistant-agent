import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from uw_course_assistant.retrieval import CatalogRetriever, create_query_planner
from uw_course_assistant.schedule_agent import create_course_agent
from uw_course_assistant.time_schedule import BrowserPool, get_time_schedule

load_dotenv()

async def main():
    print("Initializing UW Course Agent Assistant...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="db/uw_chroma_db", 
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("Loading catalog index and Cross-Encoder re-ranker...")
    catalog_retriever = CatalogRetriever(
        vectorstore=vectorstore,
        query_planner=create_query_planner(llm),
        cross_encoder=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
    )

    @tool
    def uw_course_catalog(query: str) -> str:
        """Query static UW course information. Include all of the user's hard constraints (department, credits, prerequisites, general education) in query."""
        try:
            docs = catalog_retriever.search(query)
        except ValueError as exc:
            return (
                f"Could not validate the catalog query: {exc}. "
                "Clarify the constraints; do not silently drop or relax them."
            )
        if not docs:
            return (
                "No courses matching all requested constraints were found. "
                "Do not recommend substitutes that violate those constraints."
            )
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    tools = [uw_course_catalog, get_time_schedule]

    prompt_str = "You are a professional University of Washington course selection Agent assistant.\nYou can use uw_course_catalog to query static course syllabi (including descriptions and attributes), or use get_time_schedule to query real-time schedule information for a specific quarter (such as instructors, class times, and seat availability). If the user wants to know how a course is taught in a specific quarter or if there are open seats, you MUST use get_time_schedule. If you need to combine both, you can call these two tools concurrently. Include all explicit user constraints in catalog tool queries. Recommend only courses returned by the catalog tool that satisfy those constraints. If none match, say so without relaxing constraints."

    agent_executor = create_course_agent(llm, tools, prompt=prompt_str)

    # Initialize chat history
    chat_history = []

    # --- Context Engineering: Dynamic Summarization ---
    # Threshold for triggering summarization (message count in chat_history).
    # Note: LangGraph's ReAct agent injects ToolMessages into the history,
    # so a single user turn may produce 4+ messages (Human -> AI(tool_call) -> Tool -> AI).
    # 10 messages roughly corresponds to ~2-3 full user interaction turns.
    SUMMARIZATION_THRESHOLD = 10
    # Number of recent messages to keep verbatim (preserves immediate conversational flow)
    RECENT_MESSAGES_TO_KEEP = 4

    print("✅ Agent is ready!")
    try:
        while True:
            user_input = input("\n🙋 You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            if not user_input.strip():
                continue

            try:
                chat_history.append(HumanMessage(content=user_input))
                
                # Execute the graph, handling tool calls concurrently if generated simultaneously
                response = await agent_executor.ainvoke({
                    "messages": chat_history
                })
                
                output = response["messages"][-1].content
                print(f"\n🎓 UW Agent Assistant: \n{output}")
                
                # Update chat history
                chat_history = response["messages"]

                # --- Dynamic Summarization ---
                # When history grows beyond the threshold, compress older messages
                # into a concise summary to prevent context window overflow.
                if len(chat_history) > SUMMARIZATION_THRESHOLD:
                    print("\n📝 [System] Compressing conversation history...")

                    # Partition: keep recent messages intact for conversational continuity
                    recent_messages = chat_history[-RECENT_MESSAGES_TO_KEEP:]
                    messages_to_summarize = chat_history[:-RECENT_MESSAGES_TO_KEEP]

                    # Build the summarization request
                    summary_prompt = SystemMessage(content=(
                        "You are a conversation summarizer. Distill the following messages "
                        "into a single concise summary paragraph. Focus on:\n"
                        "1. The user's specific course preferences and constraints "
                        "(e.g., department, credits, time preferences, no prerequisites).\n"
                        "2. Key facts and course recommendations already established.\n"
                        "3. Any unresolved questions the user still has.\n"
                        "IMPORTANT: If the first message is an existing summary of earlier "
                        "conversation, incorporate and UPDATE it with the new information "
                        "that follows. Do not discard prior context.\n"
                        "Do NOT include greetings or filler. Be factual and dense.\n"
                        "Keep the summary under 200 words. If the existing summary is already "
                        "long, aggressively compress older details that are no longer relevant."
                    ))

                    # Filter out ToolMessages for the summary input to reduce noise;
                    # only keep Human and AI messages which carry the semantic content.
                    semantic_messages = [
                        m for m in messages_to_summarize
                        if isinstance(m, (HumanMessage, AIMessage, SystemMessage))
                    ]
                    summary_response = await llm.ainvoke([summary_prompt] + semantic_messages)

                    # Rebuild chat_history: summary + recent raw messages
                    new_summary = SystemMessage(
                        content=f"Summary of previous conversation:\n{summary_response.content}"
                    )
                    chat_history = [new_summary] + recent_messages
                    print("✅ [System] History compressed successfully.")

            except Exception as e:
                print(f"❌ An error occurred: {e}")
    finally:
        await BrowserPool.close()

if __name__ == "__main__":
    asyncio.run(main())

