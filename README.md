# UW Course Assistant (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) assistant designed to help students discover and filter University of Washington (UW) courses using LangChain and ChromaDB.

## Features

- **Semantic Search**: Ask questions in natural language (e.g., "Recommend some easy A&H courses").
- **Self-Querying Metadata Filtering**: The LLM automatically translates natural language constraints into hard database filters (e.g., "Courses in CSE without prerequisites").
- **Local Persistence**: Course data is vectorized and persisted locally using ChromaDB to avoid repeated embedding costs.

## Requirements

- Python 3.9+
- An OpenAI API Key (`OPENAI_API_KEY`)

## Setup & Installation

1. **Clone the repository** (if you haven't already).

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   .\venv\Scripts\Activate.ps1
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

5. **Provide Course Data**:
   Ensure you have the scraped data file `courses.json` in the root directory.

## Usage

### 1. Build the Vector Database
Before you can chat with the assistant, you must embed the course data and build the local Chroma database.
```bash
python build_vector_db.py
```
*Note: If you run this again, it will automatically delete the old database folder (`./uw_chroma_db`) to prevent duplicate entries.*

### 2. Run the Course Assistant
Once the database is built, start the interactive chat application:
```bash
python app.py
```
You can type your queries into the console, and type `quit` or `exit` to stop.

## Project Structure
- `app.py`: The main interactive conversational RAG pipeline using `SelfQueryRetriever`.
- `build_vector_db.py`: Parses `courses.json`, extracts rich metadata, and stores the embeddings into ChromaDB.
- `uw_course_scraper.py`: (Optional) The original web scraper script used to gather course data from UW's catalog.
- `courses.json`: The raw JSON dataset of UW courses.
