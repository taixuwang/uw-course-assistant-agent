# Tests and evaluation

Run the commands below from the repository root. Install the project and test
dependencies in the project virtual environment first:

```bash
python3 -m venv venv
venv/bin/python -m pip install -e '.[test]'
```

## Layout

| Path | Purpose |
| --- | --- |
| `unit/` | Automated, offline tests for retrieval, context budgets, live-schedule failure handling, evaluation inputs, and the independent constraint scorer. |
| `evaluation/` | The benchmark runner, shared RAGAS support, and constraint scorer. |
| `benchmarks/` | Opt-in browser initialization microbenchmark. |
| `fixtures/` | Versioned benchmark questions and references. |
| `results/` | Saved benchmark results; record the fixture and index fingerprints when comparing runs. |

## Offline tests

```bash
venv/bin/python -m pytest -q
```

The offline suite does not make API calls, download model weights, or launch a
browser. It includes a temporary Chroma index with deterministic fake embeddings,
as well as test doubles for query planning and reranking. These tests verify
specific behavior; they do not establish quality on real user questions or
measure real-model accuracy.

Schedule tests simulate browser/network failures, bounded retries, login timeout,
parsing and embedding failures, and resource cleanup. A fake-model agent test
verifies that an unavailable schedule ends the current turn with a fixed notice,
without another model call or fabricated schedule details.

Context tests use deterministic fake token counts and summary responses. They
cover input/output reserves, system prompt and schema overhead, exact budget
boundaries, checks after tool execution, parallel and sequential tool pairing,
complete-turn summary batches, rolling summaries, and rejection of indivisible
oversized turns. They also verify that compressed messages replace graph history
and that schedule failures still produce the fixed reply. These tests validate
budget enforcement and message structure, not summary quality from a real model.

## Catalog evaluation

The evaluator uses the actual `CatalogRetriever.search()` path, including the
LLM query planner, shared filtering, dense and sparse retrieval, and reranking.
It requires:

- A configured `OPENAI_API_KEY` in the environment or repository-root `.env`.
- A built Chroma catalog index at `db/uw_chroma_db` (see the root README).
- Network access for OpenAI requests and, if not cached, the cross-encoder
  model download. Retrieval and RAGAS scoring incur API costs.

Run the current benchmark:

```bash
venv/bin/python -m tests.evaluation.eval_benchmark_v2
```

This reads `fixtures/benchmark_v2.json` and writes per-case results, summary,
model configuration, and fixture and index fingerprints to a timestamped file in
`tests/results/`. Use `--help` to see `--dataset`, `--index`, `--output`, and
`--overwrite`. Existing output is not overwritten unless `--overwrite` is
provided. Reference descriptions are checked against the index before evaluation.

The benchmark has three categories:

- **Factual:** 30 questions name a course. RAGAS context precision and recall
  evaluate retrieved contexts against that course's catalog description;
  target-course hit indicates whether the course appears among the results.
- **Discovery:** 30 topic questions each have an incomplete list of known
  relevant courses. A known-course hit is useful evidence, but a miss does not
  mean all returned courses are irrelevant.
- **Constraints:** 30 questions are assessed against independent predicates
  over the index metadata. The report includes known-eligible counts, returned
  violations and unknowns, positive-case hits, and correct empty responses.
  Unknown returned metadata is included in violations and counted separately.

These are synthetic cases, not a representative or independently reviewed set
of student questions. The factual cases explicitly include course codes; the
discovery labels are incomplete. A zero-match constraint result means there is
no *known* match in the index, not necessarily none in the real catalog. Source
metadata may itself be incorrect or missing; for example, ingestion can confuse
unknown and absent prerequisites. The scorer uses exact department metadata and
does not resolve cross-listed courses. Constraint scores do not measure topical
relevance or final agent answer quality.

The 90-question dataset extends the earlier 42-question fixture. Historical
results apply to the fixture hash recorded in each results file, not the expanded
dataset; compare scores only when they use the same questions.

## Browser microbenchmark

```bash
venv/bin/python -m tests.benchmarks.browser_latency
```

This opt-in script requires the Playwright Chromium binary (`venv/bin/python -m
playwright install chromium`). It measures browser/page initialization in a
small synthetic experiment; it does not measure full schedule retrieval,
production `BrowserPool` behavior, concurrent throughput, or end-to-end latency.
