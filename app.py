import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

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
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Failed to fetch data from {url}, error: {e}"
        
    soup = BeautifulSoup(response.content, 'html.parser')
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional University of Washington course selection Agent assistant.\nYou can use uw_course_catalog to query static course syllabi (including descriptions and attributes), or use get_time_schedule to query real-time schedule information for a specific quarter (such as instructors, class times, and seat availability). If the user wants to know how a course is taught in a specific quarter or if there are open seats, you MUST use get_time_schedule. If you need to combine both, you can call these two tools consecutively."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Initialize chat history
    chat_history = []

    print("✅ Agent is ready!")
    while True:
        user_input = input("\n🙋 You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        if not user_input.strip():
            continue

        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            print(f"\n🎓 UW Agent Assistant: \n{response['output']}")
            
            # Update chat history
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response['output'])
            ])
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()