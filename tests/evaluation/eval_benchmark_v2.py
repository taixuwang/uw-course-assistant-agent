"""Run the v2 catalog benchmark against the actual application retriever."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from tests.evaluation.benchmark_constraints import evaluate_constraint_case
from tests.evaluation.eval_ragas import (
    DEFAULT_INDEX, PROJECT_ROOT, create_catalog_retriever, evaluate_data,
)


DEFAULT_BENCHMARK = PROJECT_ROOT / "tests/fixtures/benchmark_v2.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "tests/results"


def load_cases(path):
    with Path(path).open(encoding="utf-8") as file:
        benchmark = json.load(file)
    cases = benchmark["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Benchmark must contain cases with unique IDs")
    if len({case["query"].strip().casefold() for case in cases}) != len(cases):
        raise ValueError("Benchmark contains duplicate queries")
    if any(case["category"] not in ("factual", "constraints", "discovery") for case in cases):
        raise ValueError("Unknown benchmark category")
    if {case["category"] for case in cases} != {"factual", "constraints", "discovery"}:
        raise ValueError("Benchmark needs factual, discovery, and constraint cases")
    return benchmark


def snapshot_hash(documents):
    """Fingerprint the same deduplicated text and metadata used by the retriever."""
    items = sorted((doc.metadata["course"], doc.page_content, doc.metadata) for doc in documents)
    content = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def check_reference_snapshot(cases, documents):
    """Catch changes to target descriptions before interpreting factual scores."""
    by_id = {doc.metadata["course"]: doc for doc in documents}
    for case in cases:
        doc = by_id.get(case["course_id"])
        if doc is None:
            raise ValueError(f"{case['id']}: {case['course_id']} is missing from the index")
        description = doc.page_content.partition("Description: ")[2]
        if not description or not case["reference"].endswith(description):
            raise ValueError(f"{case['id']}: reference does not match index description")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path,
                        help="Result JSON path (defaults to a timestamped file in tests/results)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing --output file")
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = DEFAULT_RESULTS_DIR / f"benchmark_v2_{stamp}.json"
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite to replace it")

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    benchmark = load_cases(args.dataset)
    cases = benchmark["cases"]
    factual = [case for case in cases if case["category"] == "factual"]
    discovery = [case for case in cases if case["category"] == "discovery"]
    constraints = [case for case in cases if case["category"] == "constraints"]
    retriever = create_catalog_retriever(args.index)
    documents = list(retriever.documents.values())
    check_reference_snapshot(factual, documents)
    for case in discovery:
        if not case["known_course_ids"] or any(code not in retriever.documents for code in case["known_course_ids"]):
            raise ValueError(f"{case['id']}: known course IDs must exist in the index")
    fingerprint = snapshot_hash(documents)
    print(f"Catalog: {len(documents)} courses; SHA-256: {fingerprint}", flush=True)

    rows = []
    ragas_data = {"user_input": [], "reference": [], "retrieved_contexts": []}
    factual_rows = []
    for number, case in enumerate(cases, start=1):
        print(f"Retrieving {number}/{len(cases)} ({case['id']}): {case['query']}", flush=True)
        docs = retriever.search(case["query"])
        ids = [doc.metadata.get("course") for doc in docs]
        row = {"id": case["id"], "category": case["category"], "query": case["query"], "returned_ids": ids}
        if case["category"] == "factual":
            row.update({"expected_course_id": case["course_id"], "target_hit": case["course_id"] in ids})
            factual_rows.append(row)
            ragas_data["user_input"].append(case["query"])
            ragas_data["reference"].append(case["reference"])
            ragas_data["retrieved_contexts"].append([doc.page_content for doc in docs])
        elif case["category"] == "discovery":
            row["known_course_ids"] = case["known_course_ids"]
            row["known_hit"] = bool(set(ids) & set(case["known_course_ids"]))
        else:
            row.update(evaluate_constraint_case(case, documents, docs))
        rows.append(row)

    print(f"Scoring {len(factual)} course-specific cases with RAGAS.", flush=True)
    result = evaluate_data(ragas_data)
    frame = result.to_pandas()
    metrics = ("llm_context_precision_with_reference", "context_recall")
    for index, row in enumerate(factual_rows):
        row["ragas"] = {}
        for metric in metrics:
            value = float(frame.iloc[index][metric])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite RAGAS score for {row['id']}: {metric}")
            row["ragas"][metric] = value

    constraint_rows = [row for row in rows if row["category"] == "constraints"]
    positive = [row for row in constraint_rows if not row["empty_expected"]]
    negative = [row for row in constraint_rows if row["empty_expected"]]
    returned = sum(row["returned_count"] for row in constraint_rows)
    violations = sum(row["violation_count"] for row in constraint_rows)
    summary = {
        "factual_cases": len(factual),
        "factual_target_hits": sum(row["target_hit"] for row in factual_rows),
        "ragas": {metric: sum(row["ragas"][metric] for row in factual_rows) / len(factual_rows) for metric in metrics},
        "discovery_cases": len(discovery),
        "discovery_known_hits": sum(row["known_hit"] for row in rows if row["category"] == "discovery"),
        "constraint_cases": len(constraints),
        "constraint_positive_cases": len(positive),
        "constraint_positive_hits": sum(row["hit"] for row in positive),
        "constraint_empty_cases": len(negative),
        "constraint_empty_correct": sum(row["empty_correct"] for row in negative),
        "constraint_returned": returned,
        "constraint_violations_including_unknown": violations,
        "constraint_unknown": sum(row["unknown_count"] for row in constraint_rows),
    }
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(args.dataset),
        "benchmark_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "index": str(args.index),
        "index_snapshot_sha256": fingerprint,
        "index_course_count": len(documents),
        "models": {"planner_and_ragas": "gpt-4o-mini", "embeddings": "text-embedding-3-small", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as file:
        file.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("Benchmark summary:", json.dumps(summary, indent=2), flush=True)
    print(f"Per-case results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
