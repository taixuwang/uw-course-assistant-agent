"""Evaluate the application's catalog retriever against reference answers.

The fixture supplies questions and answers only; retrieval uses the same persisted
catalog index and code path as the application.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "tests/fixtures/test_set_hard_60.json"
DEFAULT_INDEX = PROJECT_ROOT / "db/uw_chroma_db"


def load_benchmark(path):
    with Path(path).open(encoding="utf-8") as file:
        benchmark = json.load(file)
    queries = benchmark["queries"]
    references = benchmark["ground_truths"]
    if not isinstance(queries, list) or not isinstance(references, list):
        raise ValueError("Benchmark queries and ground_truths must be lists")
    if not queries or len(queries) != len(references):
        raise ValueError("Benchmark must contain equal, nonzero numbers of queries and ground_truths")
    return queries, references


def build_evaluation_data(retriever, queries, references):
    """Run each question through the actual retriever; preserve empty results."""
    if len(queries) != len(references):
        raise ValueError("queries and references must have the same length")
    contexts = []
    for index, query in enumerate(queries, start=1):
        print(f"Retrieving {index}/{len(queries)}: {query}", flush=True)
        docs = retriever.search(query)
        contexts.append([doc.page_content for doc in docs])
    return {"user_input": queries, "retrieved_contexts": contexts, "reference": references}


def create_catalog_retriever(index_path):
    # Check before opening Chroma: it otherwise silently creates an empty database.
    index_path = Path(index_path)
    if not (index_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"Catalog index not found at {index_path}. "
            "Build it first with python scripts/build_vector_db.py from the project root."
        )

    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from sentence_transformers import CrossEncoder
    from uw_course_assistant.retrieval import CatalogRetriever, create_query_planner

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=str(index_path), embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return CatalogRetriever(
        vectorstore=vectorstore,
        query_planner=create_query_planner(llm),
        cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )


def evaluate_data(data):
    """Score previously retrieved contexts without running retrieval again."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI

    # RAGAS 0.4.x imports this removed optional Vertex AI integration even
    # when using OpenAI. Restore the legacy compatibility shim for that case.
    if importlib.util.find_spec("langchain_community.chat_models.vertexai") is None:
        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module.__name__] = module
    from ragas import evaluate

    try:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
    except ImportError:
        from ragas.metrics import context_precision, context_recall
        metrics = [context_precision, context_recall]

    return evaluate(
        dataset=Dataset.from_dict(data),
        metrics=metrics,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        raise_exceptions=True,
    )


def evaluate_retriever(retriever, queries, references):
    data = build_evaluation_data(retriever, queries, references)
    empty_count = sum(not contexts for contexts in data["retrieved_contexts"])
    print(f"Retrieval returned no contexts for {empty_count}/{len(queries)} queries.")
    return evaluate_data(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="JSON fixture containing queries and ground_truths")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                        help="Persisted production-format Chroma index")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    queries, references = load_benchmark(args.dataset)
    print(f"Evaluating {len(queries)} queries from {args.dataset} against {args.index}.")
    print("The fixture corpus is not used; both retrieval branches use the catalog index.")
    retriever = create_catalog_retriever(args.index)
    print(f"Catalog snapshot contains {len(retriever.documents)} distinct courses.")
    results = evaluate_retriever(retriever, queries, references)
    print("Production catalog retriever RAGAS results:")
    print(results)


if __name__ == "__main__":
    main()
