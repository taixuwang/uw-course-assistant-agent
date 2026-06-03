# build_vector_db.py
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load environment variables from .env file (gets OPENAI_API_KEY)
load_dotenv() 
# Make sure to run pip install python-dotenv first

def main():
    print("Start loading course data...")
    
    # 2. Read the actual scraped data
    # Adjust the key names here based on your actual json file structure
    documents = []
    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except FileNotFoundError:
        print("Error: courses.json file not found, please check the path.")
        return

    # 3. Convert JSON data to LangChain Document objects
    for course in tqdm(course_data, desc="Parsing course data"):
        # Note: replace .get('xxx') with the actual key names from your JSON file
        # For example, if your scraper stores "course_id", write course.get("course_id") here
        gen_ed = course.get('gen_ed')
        gen_ed_str = ", ".join(gen_ed) if gen_ed else 'N/A'
        prereq = course.get('prerequisites') or 'None'
        
        page_content = f"""Course: {course.get('course', 'N/A')} - {course.get('title', 'N/A')}
Credits: {course.get('credits', 'N/A')}
General Education: {gen_ed_str}
Prerequisites: {prereq}
Description: {course.get('description', 'N/A')}"""
        
        # Store the extracted structured fields into metadata to enable precise hard filtering by the LLM
        course_code = course.get("course") or "unknown"
        department = course_code.split(" ")[0] if " " in course_code else "unknown"
        prereq_str = str(prereq) if prereq else "None"
        has_prerequisite = bool(prereq and prereq_str.lower() != 'none')

        metadata = {
            "course": course_code,
            "department": department,
            "has_prerequisite": has_prerequisite,
            "title": str(course.get("title") or "N/A"),
            "credits": str(course.get("credits") or "N/A"),
            "gen_ed": gen_ed_str,
            "prerequisites": prereq_str
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Successfully loaded {len(documents)} courses.")
    print("Calling OpenAI Embedding model and batch inserting into ChromaDB (may take a few minutes)...")

    # 4. If an old database exists, delete it first to prevent duplicate data appending
    if os.path.exists("./uw_chroma_db"):
        print("Detected old database, cleaning up to avoid data duplication...")
        import shutil
        shutil.rmtree("./uw_chroma_db")

    # 5. Vectorize and persist storage
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # This will generate a uw_chroma_db folder in your RAG directory
    # Use tqdm for batch processing to show a progress bar and avoid large single requests
    vectorstore = Chroma(persist_directory="./uw_chroma_db", embedding_function=embeddings)
    
    batch_size = 500
    for i in tqdm(range(0, len(documents), batch_size), desc="Vectorizing and inserting data"):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("✅ Offline knowledge base construction completed and persisted to ./uw_chroma_db directory!")

if __name__ == "__main__":
    main()