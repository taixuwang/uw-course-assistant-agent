import asyncio
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from uw_course_assistant.context import ContextBudget
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

    context_budget = ContextBudget(
        context_window_tokens=int(os.getenv("CONTEXT_WINDOW_TOKENS", "16000")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),
        safety_margin_tokens=int(os.getenv("CONTEXT_SAFETY_MARGIN_TOKENS", "512")),
        summary_tokens=int(os.getenv("CONTEXT_SUMMARY_TOKENS", "512")),
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=context_budget.max_output_tokens)

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

    agent_executor = create_course_agent(llm, tools, prompt=prompt_str, context_budget=context_budget)

    # Initialize chat history
    chat_history = []

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
                turn_messages = [*chat_history, HumanMessage(content=user_input)]
                
                # Execute the graph, handling tool calls concurrently if generated simultaneously
                response = await agent_executor.ainvoke({
                    "messages": turn_messages
                })
                
                output = response["messages"][-1].content
                print(f"\n🎓 UW Agent Assistant: \n{output}")
                
                # Update chat history
                chat_history = response["messages"]

            except Exception as e:
                print(f"❌ An error occurred: {e}")
    finally:
        await BrowserPool.close()

if __name__ == "__main__":
    asyncio.run(main())

