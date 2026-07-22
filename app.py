import os
import json
import requests
import asyncio
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# --- Async Playwright Pool Singleton ---
class BrowserPool:
    _playwright = None
    _browser = None

    @classmethod
    async def get_browser(cls):
        if cls._playwright is None:
            cls._playwright = await async_playwright().start()
        if cls._browser is None or not cls._browser.is_connected():
            cls._browser = await cls._playwright.chromium.launch(headless=False)
        return cls._browser

    @classmethod
    async def close(cls):
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright:
            await cls._playwright.stop()
            cls._playwright = None

@tool
async def get_time_schedule(quarter_year: str, department: str, query: str) -> str:
    """
    Call this tool when you need to query real-time course schedule information (including specific class times, locations, instructors, availability, etc.) for a specific quarter (e.g., 'SPR2024') and department (e.g., 'cse').
    Input parameters:
    - quarter_year: Quarter and year, e.g., 'AUT2024', 'WIN2024', 'SPR2024', 'SUM2024'
    - department: Department code, e.g., 'cse', 'math'
    - query: The user's specific question, e.g., 'Which sections of CSE 121 are Open?' or 'What is the class time for CSE 143?'
    """
    url = f"https://www.washington.edu/students/timeschd/{quarter_year.upper()}/{department.lower()}.html"
    
    try:
        browser = await BrowserPool.get_browser()
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(url)
            
            # Check if redirected to a login page or UW NetID authentication
            current_url = page.url.lower()
            if "login" in current_url or "signin" in current_url or "identity" in current_url or "shibboleth" in current_url:
                print(f"\n⚠️ Login required. Please complete UW NetID login and 2FA in the popped-up browser window...")
                print(f"⏳ Waiting for you to log in and be redirected back to the schedule page (up to 5 minutes)...")
                
                # Wait for the URL to change back to the intended quarter/department page.
                # using a 5-minute timeout (300000 ms)
                await page.wait_for_url(f"**/{quarter_year.upper()}/**", timeout=300000)
                
            # Wait for the course table to ensure page is loaded
            try:
                await page.wait_for_selector("table[bgcolor='#ccffcc']", timeout=10000)
            except Exception:
                pass # If it times out, we will just pass the current content to BeautifulSoup to let it handle "no courses found"
                
            html_content = await page.content()
        finally:
            await page.close() # Close tab page, keep browser in pool alive
            
    except Exception as e:
        return f"Failed to fetch data from {url}, error: {e}"
        
    soup = BeautifulSoup(html_content, 'html.parser')
    course_tables = soup.find_all('table', bgcolor='#ccffcc')
    
    documents = []
    for c_table in course_tables:
        # Get the course Header (e.g., CSE 121 COMP PROGRAMMING I)
        course_header = c_table.get_text(separator=" ", strip=True)
        
        # The series of tables immediately following this Header contain the Section information, until the next colored Header
        curr = c_table.find_next_sibling('table')
        sections_text = []
        while curr and curr.get('bgcolor') != '#ccffcc':
            pre = curr.find('pre')
            if pre:
                # Extract the plain text formatting from <pre>
                text = pre.get_text(separator="  ", strip=True)
                sections_text.append(text)
            curr = curr.find_next_sibling('table')
            
        if sections_text:
            # Concatenate all Sections under the same course into a single Document for the LLM to read uniformly
            page_content = f"Course Info: {course_header}\nSections:\n" + "\n---\n".join(sections_text)
            documents.append(Document(page_content=page_content))
            
    if not documents:
        return f"No structured course information found at {url}. Please check if the link is valid."
        
    # Create a temporary vector database to retrieve current webpage information
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents, embeddings)
    # k=10, fetch data for the 10 most relevant courses
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant course sections found."
        
    # Simply return the raw text of the retrieved documents.
    # The main Agent's LLM will read this string (as the "Observation") and generate the final answer.
    return "\n\n".join([doc.page_content for doc in docs])

async def main():
    print("Initializing UW Course Agent Assistant...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./uw_chroma_db", 
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    metadata_field_info = [
        AttributeInfo(name="course", description="Course code, e.g., 'CSE 374'", type="string"),
        AttributeInfo(name="department", description="Department code, e.g., 'CSE'", type="string"),
        AttributeInfo(name="has_prerequisite", description="True if has prerequisites", type="boolean"),
        AttributeInfo(name="title", description="Course title", type="string"),
        AttributeInfo(name="credits", description="Credits", type="string"),
        AttributeInfo(name="gen_ed", description="General Education requirements", type="string"),
        AttributeInfo(name="prerequisites", description="Prerequisites text", type="string")
    ]
    
    document_content_description = "Detailed information about University of Washington courses."

    # 1. Self-Query Dense Vector Retriever
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        enable_limit=True,
        search_kwargs={"k": 10}
    )

    # 2. BM25 Sparse Retriever (Load catalog documents)
    print("Building BM25 sparse index...")
    all_docs = []
    if os.path.exists("courses.json"):
        with open("courses.json", "r", encoding="utf-8") as f:
            course_data = json.load(f)
        for course in course_data:
            code = course.get("course") or "unknown"
            title = course.get("title") or "N/A"
            credits_str = str(course.get("credits") or "N/A")
            description = course.get("description") or "N/A"
            gen_ed_list = course.get("gen_ed")
            gen_ed_str = ", ".join(gen_ed_list) if isinstance(gen_ed_list, list) else str(gen_ed_list or "N/A")
            prereq = course.get("prerequisites")
            prereq_str = str(prereq) if prereq else "None"
            page_content = f"Course: {code} - {title}\nCredits: {credits_str}\nGeneral Education: {gen_ed_str}\nPrerequisites: {prereq_str}\nDescription: {description}"
            metadata = {
                "course": code,
                "department": code.split(" ")[0] if " " in code else "unknown",
                "has_prerequisite": bool(prereq and prereq_str.lower() != 'none'),
                "title": title,
                "credits": credits_str,
                "gen_ed": gen_ed_str,
                "prerequisites": prereq_str
            }
            all_docs.append(Document(page_content=page_content, metadata=metadata))
    
    bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
    if bm25_retriever:
        bm25_retriever.k = 10

    # 3. Cross-Encoder Re-ranker
    print("Loading Cross-Encoder re-ranker model...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 4. Define Hybrid + Self-Query + Re-ranking RAG Tool
    @tool
    def uw_course_catalog(query: str) -> str:
        """Use this tool when the user asks for general introductions, prerequisites, credits, general education attributes, or other static knowledge base information about university courses."""
        # Step A: Dense Retrieval with Self-Query Metadata Filter
        dense_docs = self_query_retriever.invoke(query)
        
        # Step B: BM25 Sparse Keyword Retrieval
        bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
        
        # Step C: Reciprocal Rank Fusion (RRF)
        k_constant = 60
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k_constant + rank + 1)
            
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k_constant + rank + 1)
            
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc_map[doc_id] for doc_id, _ in sorted_docs[:10]]
        
        if not candidates:
            return "No relevant course catalog information found."
            
        # Step D: Cross-Encoder Re-ranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = cross_encoder.predict(pairs)
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, _ in scored_docs[:5]]
        return "\n\n---\n\n".join([doc.page_content for doc in top_docs])

    tools = [uw_course_catalog, get_time_schedule]

    prompt_str = "You are a professional University of Washington course selection Agent assistant.\nYou can use uw_course_catalog to query static course syllabi (including descriptions and attributes), or use get_time_schedule to query real-time schedule information for a specific quarter (such as instructors, class times, and seat availability). If the user wants to know how a course is taught in a specific quarter or if there are open seats, you MUST use get_time_schedule. If you need to combine both, you can call these two tools concurrently."

    agent_executor = create_react_agent(llm, tools, prompt=prompt_str)

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
