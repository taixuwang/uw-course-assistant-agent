# UW Course Agent Assistant

An intelligent, autonomous Agent designed to help students discover, filter, and plan University of Washington (UW) courses. Built with LangChain, OpenAI (`gpt-4o-mini`), and ChromaDB, this assistant leverages a ReAct tool-calling architecture to dynamically retrieve both static course syllabi and real-time schedule information.

## Key Features

- **ReAct Agent Architecture**: The LLM autonomously decides which tool to use based on your question, allowing it to seamlessly switch between querying a static knowledge base and fetching live web data.
- **Conversational Memory**: The Agent remembers your previous questions, allowing for natural, multi-turn follow-up conversations (e.g., "Find me a CSE course without prerequisites" -> "What time is that course taught this Spring?").
- **Multi-Tool Integration**:
  - `uw_course_catalog` (Self-Query Retriever RAG): Automatically translates natural language into hard database filters (e.g., department, prerequisites, credits) to search offline course data.
  - `get_time_schedule` (Live Scraper): Dynamically scrapes UW's Time Schedule webpage for a specific quarter and department, providing real-time data on section times, locations, instructors, and seat availability (Open/Closed).

## Requirements

- Python 3.9+
- An OpenAI API Key (`OPENAI_API_KEY`)
- Required packages: `langchain`, `langchain-openai`, `langchain-chroma`, `beautifulsoup4`, `requests`, `python-dotenv`

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
   Ensure `requests` and `beautifulsoup4` are included in your environment alongside LangChain.
   ```bash
   pip install -r requirements.txt
   pip install requests beautifulsoup4
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
Before you can chat with the assistant, you must embed the static course data and build the local Chroma database.
```bash
python build_vector_db.py
```
*Note: If you run this again, it will automatically delete the old database folder (`./uw_chroma_db`) to prevent duplicate entries.*

### 2. Run the Course Agent
Once the database is built, start the interactive chat application:
```bash
python app.py
```
You can type your queries into the console. Try combining static and dynamic requests, such as: 
> *"What are some introductory A&H courses? Are any of them open in SPR2024?"*

Type `quit` or `exit` to stop.

## Project Structure
- `app.py`: The main Agent application. Contains the `AgentExecutor`, conversation memory loop, and the definitions for the `get_time_schedule` and `uw_course_catalog` tools.
- `build_vector_db.py`: Parses `courses.json`, extracts rich metadata, and stores the static embeddings into ChromaDB.
- `uw_course_scraper.py`: (Optional) The original web scraper script used to gather course data from UW's catalog.
- `courses.json`: The raw JSON dataset of UW courses.
