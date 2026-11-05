"""Shared catalog retriever setup and RAGAS scoring for evaluations."""

import importlib.util
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = PROJECT_ROOT / "db/uw_chroma_db"


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
