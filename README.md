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
- **Live Schedule Scraping via Playwright** — An async browser pool singleton (`BrowserPool`) keeps a long-lived Chromium instance for low-latency page loads. Supports interactive UW NetID / 2FA login in a browser pop-up. Transient failures receive up to three attempts with 1- and 2-second backoff; invalid arguments, permanent HTTP errors, and login timeout stop early. Each attempt closes its browser context, and retrieval uses an isolated temporary index.
- **Explicit Live-Data Failures** — Browser, parsing, embedding, and retrieval failures return a structured unavailable result. If any schedule call fails in the current turn, the agent ends that turn with a fixed unavailable notice and official source links, without asking the LLM to generate schedule details or retry again. A later user message can try again. Unavailable data is never treated as evidence that there are no courses or open seats.
- **Token-Budgeted Conversational Memory** — Checks the system prompt, tool schemas, and message history before every agent model call, including after tools return, with reserved output tokens and a safety margin. Compresses older complete user turns into a rolling summary while preserving the entire current turn and all tool-call/result pairs.

## Multi-Tool Integration

| Tool | Source | Description |
|---|---|---|
| `uw_course_catalog` | Offline ChromaDB + BM25 | Hybrid retrieval over the full UW course catalog with cross-encoder re-ranking. Handles queries about descriptions, prerequisites, credits, and general education attributes. |
| `get_time_schedule` | Live UW Time Schedule | Scrapes UW's official Time Schedule pages via Playwright for a specific quarter and department. Returns section times, locations, instructors, and seat availability (Open/Closed). |

## Requirements

- Python 3.10+
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
   pip install -e .
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
   Ensure `data/courses.json` is present. You can generate it by running the scraper (see below).

## Usage

### 1. (Optional) Scrape Course Catalog

Crawl the UW course catalog and produce `data/courses.json`:

```bash
python scripts/uw_course_scraper.py
```

### 2. Build the Vector Database

Embed the course data into a local ChromaDB instance:

```bash
python scripts/build_vector_db.py
```

> Re-running this command automatically deletes the old `db/uw_chroma_db` directory to prevent duplicate entries.

### 3. Run the Agent

Start the interactive chat session:

```bash
python -m uw_course_assistant.app
```

Example queries:

> *"What are some introductory A&H courses with no prerequisites?"*
>
> *"Is CSE 121 open in SPR2024?"*
>
> *"Find me a 5-credit MATH course, then check if it's available this Autumn."*

Type `quit` or `exit` to stop.

Context budgets can be configured in `.env`:

| Variable | Default | Purpose |
|---|---:|---|
| `CONTEXT_WINDOW_TOKENS` | 16000 | Total working budget for input, output, and safety margin; keep this within the model's supported context window. |
| `MAX_OUTPUT_TOKENS` | 2048 | Reserved and enforced maximum output tokens per agent model call. |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | 512 | Headroom for provider-specific token framing and estimation differences. |
| `CONTEXT_SUMMARY_TOKENS` | 512 | Maximum output tokens for the rolling summary. |

The default input allowance is `16000 - 2048 - 512 = 13440` tokens, including
the system prompt and tool schemas. Counts use the model's local tokenizer;
tool schemas are estimated separately because the message counter does not count
them. These are estimates rather than exact provider usage. When the input exceeds
the allowance, the agent retains the newest complete turns that fit, aiming for
75% of the input allowance after reserving summary space. The current user turn
is always kept intact, including parallel tools and repeated tool calls.

Older turns, including tool contents and result status, are summarized as a plain
transcript, so the summarizer never receives dangling structured tool calls.
Summarization batches also respect input/output budgets and keep each older turn
whole. A prior summary is updated rather than accumulated. The compressed history
replaces the graph's history and is used by later turns; it remains in memory
only and is cleared on restart.

If the current turn cannot fit, a complete older turn cannot fit in a summary
request, or the summarizer fails its output checks, the request stops with a
clear error instead of sending an oversized prompt or splitting tool pairs.
The CLI retains the last successful history so another request can be tried.

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

Run the offline regression suite from the project root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The suite uses test doubles and a temporary Chroma database; it does not call
external APIs, download models, or launch a browser. To run the opt-in catalog
benchmark against a built index and configured OpenAI API key:

```bash
python -m tests.evaluation.eval_benchmark_v2
```

The benchmark makes paid API calls and may download cross-encoder weights. It
reports factual retrieval, discovery, and multi-answer constraint cases
separately; its synthetic questions and metadata-based labels have important
limitations. See [tests/README.md](tests/README.md) for the test layout,
benchmark interpretation, prerequisites, and output options.

## Project Structure

```
uw-course-assistant-agent/
├── src/
│   └── uw_course_assistant/
│       ├── app.py                  # Main application and conversation loop
│       ├── context.py              # Token budgets and complete-turn summarization
│       ├── retrieval.py            # Static catalog retrieval and constraint filtering
│       ├── time_schedule.py        # Live schedule scraping, retries, and cleanup
│       └── schedule_agent.py       # ReAct agent and deterministic unavailable response
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
    ├── unit/                       # Offline automated tests
    ├── evaluation/                 # Evaluation runner and shared scoring helpers
    ├── benchmarks/                 # Opt-in browser lifecycle microbenchmark
    ├── fixtures/                   # Benchmark data
    ├── results/                    # Saved benchmark runs
    └── README.md                   # Commands and interpretation
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
