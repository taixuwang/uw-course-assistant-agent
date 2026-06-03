# app.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (Ensure OPENAI_API_KEY is in .env)
load_dotenv()

def main():
    print("Initializing UW Course Assistant...")

    # 2. Connect to the persisted local Chroma database
    # Must use the exact same Embedding model as used during offline construction
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./uw_chroma_db", 
        embedding_function=embeddings
    )

    # 3. Initialize the LLM for routing control and final answering
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 4. Configure Self-Query Retriever (Core magic)
    # Here we need to tell the LLM what Metadata is stored in our database, so it can decide whether to perform hard filtering
    metadata_field_info = [
        AttributeInfo(
            name="course",
            description="The unique code for the course, e.g., 'CSE 374', 'MATH 208', 'ESS 101'.",
            type="string",
        ),
        AttributeInfo(
            name="department",
            description="The department code for the course, e.g., 'CSE', 'MATH', 'INFO', 'ENGL'. Use this to filter courses by department.",
            type="string",
        ),
        AttributeInfo(
            name="has_prerequisite",
            description="True if the course has prerequisites, False if it has no prerequisites. Use this to filter out courses that require prior knowledge.",
            type="boolean",
        ),
        AttributeInfo(
            name="title",
            description="The title of the course, e.g., 'Introduction to Algorithms'.",
            type="string",
        ),
        AttributeInfo(
            name="credits",
            description="The number of credits the course is worth, e.g., '5' or '1-2, max. 12'.",
            type="string",
        ),
        AttributeInfo(
            name="gen_ed",
            description="The General Education requirements satisfied by the course, e.g., 'SSc, DIV' or 'N/A'. You can use the 'contain' operator to filter by specific requirements.",
            type="string",
        ),
        AttributeInfo(
            name="prerequisites",
            description="The exact text of the prerequisites for the course, or 'None'.",
            type="string",
        )
    ]
    
    document_content_description = "Detailed information about University of Washington courses, including course name, credits, General Education requirements, Prerequisites, and course description."

    # Create the smart retriever
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        enable_limit=True, # Allow the LLM to handle limit queries like "Recommend me 3 courses"
        search_kwargs={"k": 10} # Retrieve up to 10 courses as context by default
    )

    # 5. Define the RAG System Prompt
    template = """
    You are a professional University of Washington (UW) course selection assistant. Please carefully read the following retrieved course information (Context) and answer the user's question based ONLY on this information.
    If the provided context does not contain relevant information to answer the question, honestly state "Based on the knowledge base, I don't know the answer right now", and absolutely do not fabricate course or credit information.

    [Context]
    {context}

    [User Question]
    {question}

    Please provide your answer clearly and professionally:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Helper function: format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Assemble the LCEL QA chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Assistant is ready! Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)

    # 8. Start the interactive command line conversation loop
    while True:
        user_input = input("\n🙋 You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
            
        if not user_input.strip():
            continue

        try:
            print("🤖 Thinking and retrieving...\n")
            # Call the RAG Chain
            response = rag_chain.invoke(user_input)
            print(f"🎓 UW Course Assistant: \n{response}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()