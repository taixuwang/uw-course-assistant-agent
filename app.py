import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

@tool
def get_time_schedule(quarter_year: str, department: str, query: str) -> str:
    """
    Call this tool when you need to query real-time course schedule information (including specific class times, locations, instructors, availability, etc.) for a specific quarter (e.g., 'SPR2024') and department (e.g., 'cse').
    Input parameters:
    - quarter_year: Quarter and year, e.g., 'AUT2024', 'WIN2024', 'SPR2024', 'SUM2024'
    - department: Department code, e.g., 'cse', 'math'
    - query: The user's specific question, e.g., 'Which sections of CSE 121 are Open?' or 'What is the class time for CSE 143?'
    """
    url = f"https://www.washington.edu/students/timeschd/{quarter_year.upper()}/{department.lower()}.html"
    
    try:
        with sync_playwright() as p:
            # headless=False so the user can interact with the browser if a login page appears
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            
            page.goto(url)
            
            # Check if redirected to a login page or UW NetID authentication
            current_url = page.url.lower()
            if "login" in current_url or "signin" in current_url or "identity" in current_url or "shibboleth" in current_url:
                print(f"\n⚠️ Login required. Please complete UW NetID login and 2FA in the popped-up browser window...")
                print(f"⏳ Waiting for you to log in and be redirected back to the schedule page (up to 5 minutes)...")
                
                # Wait for the URL to change back to the intended quarter/department page.
                # using a 5-minute timeout (300000 ms)
                page.wait_for_url(f"**/{quarter_year.upper()}/**", timeout=300000)
                
            # Wait for the course table to ensure page is loaded
            try:
                page.wait_for_selector("table[bgcolor='#ccffcc']", timeout=10000)
            except Exception:
                pass # If it times out, we will just pass the current content to BeautifulSoup to let it handle "no courses found"
                
            html_content = page.content()
            browser.close()
            
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

def main():
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

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        enable_limit=True,
        search_kwargs={"k": 10}
    )

    course_retriever_tool = create_retriever_tool(
        retriever,
        name="uw_course_catalog",
        description="Use this tool when the user asks for general introductions, prerequisites, credits, general education attributes, or other static knowledge base information about university courses."
    )

    tools = [course_retriever_tool, get_time_schedule]

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
            response = agent_executor.invoke({
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
                summary_response = llm.invoke([summary_prompt] + semantic_messages)

                # Rebuild chat_history: summary + recent raw messages
                new_summary = SystemMessage(
                    content=f"Summary of previous conversation:\n{summary_response.content}"
                )
                chat_history = [new_summary] + recent_messages
                print("✅ [System] History compressed successfully.")

        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()