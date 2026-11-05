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
| `unit/` | Automated, offline tests for retrieval, evaluation inputs, and the independent constraint scorer. |
| `evaluation/` | RAGAS and v2 benchmark runners, constraint scorer, and legacy fixture generators. |
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

## Catalog evaluation

The evaluators use the actual `CatalogRetriever.search()` path, including the
LLM query planner, shared filtering, dense and sparse retrieval, and reranking.
Both require:

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

- **Factual:** 20 questions name a course. RAGAS context precision and recall
  evaluate retrieved contexts against that course's catalog description;
  target-course hit indicates whether the course appears among the results.
- **Discovery:** 10 topic questions each have an incomplete list of known
  relevant courses. A known-course hit is useful evidence, but a miss does not
  mean all returned courses are irrelevant.
- **Constraints:** 12 questions are assessed against independent predicates
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

For historical comparison only, the legacy benchmark can also be run:

```bash
venv/bin/python -m tests.evaluation.eval_ragas
```

It reads `fixtures/test_set_hard_60.json` questions and references, but retrieves
from the persisted catalog index, not from the fixture's text-only corpus.
The legacy fixture has duplicate questions paired with different single-course
references and concentrates on two departments; its references may also come
from an older snapshot. Its RAGAS scores are not directly comparable with v2.

## Legacy fixture generators

These generators produce synthetic, unreviewed data from `data/courses.json`;
they do not build an index and do not regenerate the v2 benchmark. The raw
catalog is not tracked in Git. Both commands support `--input`, `--output`, and
`--force`; the hard generator also accepts `--seed`. By default they refuse to
overwrite an existing fixture. For example:

```bash
venv/bin/python -m tests.evaluation.build_test_set_60 --output /tmp/basic-benchmark.json
venv/bin/python -m tests.evaluation.build_hard_benchmark --output /tmp/hard-benchmark.json --seed 42
```

Use `--help` for defaults and options. The legacy construction is template-based;
its reference for a broad question should not be interpreted as the only valid
answer.

## Browser microbenchmark

```bash
venv/bin/python -m tests.benchmarks.browser_latency
```

This opt-in script requires the Playwright Chromium binary (`venv/bin/python -m
playwright install chromium`). It measures browser/page initialization in a
small synthetic experiment; it does not measure full schedule retrieval,
production `BrowserPool` behavior, concurrent throughput, or end-to-end latency.
