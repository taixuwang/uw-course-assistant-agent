# UW Course Assistant Agent

An intelligent, autonomous agent that helps students discover, filter, and plan University of Washington (UW) courses. Built with the LangChain ecosystem (LangGraph, OpenAI GPT-4o-mini, and ChromaDB), this assistant uses a ReAct tool-calling architecture to dynamically retrieve both static course catalog data and real-time schedule information.

## Key Features

- **ReAct Agent Architecture** — The LLM autonomously selects tools based on user queries, seamlessly switching between a static knowledge base and live web scraping.
- **Concurrent Tool Execution** — Powered by LangGraph's `create_react_agent`, the assistant can invoke multiple tools in parallel (e.g., searching catalog data while simultaneously scraping schedule pages), reducing response latency.
- **Hybrid RAG Pipeline with Shared Filtering**:
  1. **Query Planning & Validation** — Parses the question once into a relevance query and typed hard constraints. Invalid constraints fail explicitly instead of being silently removed.
  2. **Shared Eligibility Filter** — Applies constraints to a catalog snapshot from ChromaDB, producing one eligible course ID set before retrieval.
  3. **Dense + BM25 Retrieval** — Both branches rank only eligible courses, returning up to 20 candidates each. BM25 reuses corpus-wide IDF statistics while scoring only eligible courses.
  4. **Reciprocal Rank Fusion (RRF)** — Deduplicates by course ID and retains up to 20 fused candidates.
  5. **Cross-Encoder Re-Ranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — Scores candidates against the original question. A final eligibility check returns up to five courses, respecting smaller requested limits.
- **Live Schedule Scraping via Playwright** — An async browser pool singleton (`BrowserPool`) keeps a long-lived Chromium instance for low-latency page loads. Supports automatic UW NetID / 2FA login with interactive browser pop-up.
- **Conversational Memory with Dynamic Summarization** — Maintains multi-turn context with a sliding-window summarization strategy. When chat history exceeds a configurable threshold, older messages are compressed into a concise summary to prevent context window overflow.

## Multi-Tool Integration

| Tool | Source | Description |
|---|---|---|
| `uw_course_catalog` | Offline ChromaDB + BM25 | Hybrid retrieval over the full UW course catalog with cross-encoder re-ranking. Handles queries about descriptions, prerequisites, credits, and general education attributes. |
| `get_time_schedule` | Live UW Time Schedule | Scrapes UW's official Time Schedule pages via Playwright for a specific quarter and department. Returns section times, locations, instructors, and seat availability (Open/Closed). |

## Requirements

- Python 3.9+
- An OpenAI API key (`OPENAI_API_KEY`)
- Playwright browsers installed (`playwright install chromium`)

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/uw-course-assistant-agent.git
   cd uw-course-assistant-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browser binaries:**
   ```bash
   playwright install chromium
   ```

5. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

6. **Provide course data:**
   Ensure `courses.json` is present in the project root. You can generate it by running the scraper (see below).

## Usage

### 1. (Optional) Scrape Course Catalog

Crawl the UW course catalog and produce `courses.json`:

```bash
python uw_course_scraper.py
```

### 2. Build the Vector Database

Embed the course data into a local ChromaDB instance:

```bash
python build_vector_db.py
```

> Re-running this command automatically deletes the old `./uw_chroma_db` directory to prevent duplicate entries.

### 3. Run the Agent

Start the interactive chat session:

```bash
python app.py
```

Example queries:

> *"What are some introductory A&H courses with no prerequisites?"*
>
> *"Is CSE 121 open in SPR2024?"*
>
> *"Find me a 5-credit MATH course, then check if it's available this Autumn."*

Type `quit` or `exit` to stop.

Catalog retrieval uses the persisted vector database as the common source of text
and metadata for both branches, so this change does not require rebuilding an
existing index. If the catalog data changes, rebuild the index as usual.
Fixed-credit filters are numeric: `5` and `5, max. 15` match a five-credit filter;
variable-credit entries such as `1-5` or `3/5` and unknown credits do not match
numeric credit filters. General education filters match individual codes, including
codes separated by `/`. Missing metadata does not satisfy a constraint, including
negated constraints. No matching courses produces an empty result without relaxing
the user's constraints. Eligibility remains dependent on the accuracy of indexed
course metadata and the LLM's extraction of the user's constraints.

## Evaluation & Testing

Run the shared-filter retrieval regression tests without API calls, model downloads,
or browser launches (after installing the project dependencies):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_retrieval.py -v
```

These tests cover filter validation, numeric credits, general education membership,
logical conditions, shared eligibility before top-k selection, course ID deduplication,
empty results, and ID restriction with a temporary real Chroma database. The older
RAGAS script below evaluates its own retrieval implementation; it does not exercise
the production query planner or shared eligibility filter.

The `test/` directory contains a RAGAS-based evaluation suite and latency benchmarks:

| File | Description |
|---|---|
| `test/build_test_set_60.py` | Generates a 60-query test set from `courses.json` with verified ground-truth answers. |
| `test/build_hard_benchmark.py` | Generates a harder 60-query benchmark with a 1,000-document distractor corpus (5 query categories: topic/skill, prerequisite, gen-ed/credit, concept overlap, and disambiguation). |
| `test/eval_ragas.py` | Runs RAGAS evaluation (Context Precision & Context Recall) on the final hybrid + re-ranker retriever pipeline. |
| `test/test_latency.py` | Benchmarks Playwright page-initialization latency: cold browser launch vs. async browser pool singleton. |

Run the evaluation:

```bash
cd test
python eval_ragas.py
```

## Project Structure

```
uw-course-assistant-agent/
├── src/
│   └── uw_course_assistant/
│       └── app.py                  # Main agent application (ReAct agent, tools, conversation loop)
├── scripts/
│   ├── build_vector_db.py          # Parses courses.json → ChromaDB vector store
│   └── uw_course_scraper.py        # Web scraper for UW course catalog → courses.json
├── data/
│   └── courses.json                # Raw course dataset (scraped from UW catalog)
├── db/
│   └── uw_chroma_db/               # Persistent Chroma vector store
├── pyproject.toml                  # Python project metadata and dependencies
├── .env                            # Environment variables (not tracked)
└── tests/
    ├── eval_ragas.py               # RAGAS retriever evaluation script
    ├── test_latency.py             # Playwright browser pool latency benchmark
    ├── build_test_set_60.py        # Test set generator (60 basic Q&A pairs)
    ├── build_hard_benchmark.py     # Hard benchmark generator (60 queries + 1K distractors)
    └── fixtures/
        ├── test_set_60.json        # Pre-built test set (60 pairs)
        └── test_set_hard_60.json   # Pre-built hard benchmark dataset
```

## Tech Stack

- **Agent Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) (ReAct agent)
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Store:** [ChromaDB](https://www.trychroma.com/)
- **Sparse Retrieval:** BM25 (via `rank_bm25`)
- **Re-Ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence-transformers`)
- **Web Scraping:** [Playwright](https://playwright.dev/python/) (async) + BeautifulSoup
- **Evaluation:** [RAGAS](https://docs.ragas.io/)

## License

This project is for educational purposes.
