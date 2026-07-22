import os
import sys
import types

# Polyfill for missing vertexai module in newer langchain_community
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = m

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Ragas core modules
from ragas import evaluate
try:
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    metric_precision = LLMContextPrecisionWithReference()
    metric_recall = LLMContextRecall()
except ImportError:
    from ragas.metrics import context_precision, context_recall
    metric_precision = context_precision
    metric_recall = context_recall

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Retriever libraries
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

import json

# ==========================================
# 1. Load 60 verified benchmark Q&A test pairs
# ==========================================
test_set_file = "test_set_60.json"
if os.path.exists(test_set_file):
    with open(test_set_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    raw_documents = test_data["documents"]
    test_queries = test_data["queries"]
    ground_truths = test_data["ground_truths"]
    sample_docs = [Document(page_content=doc_str) for doc_str in raw_documents]
    print(f"[Loaded] {len(test_queries)} verified Q&A test pairs from {test_set_file}.")
elif os.path.exists("test_set_30.json"):
    with open("test_set_30.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    raw_documents = test_data["documents"]
    test_queries = test_data["queries"]
    ground_truths = test_data["ground_truths"]
    sample_docs = [Document(page_content=doc_str) for doc_str in raw_documents]
    print(f"[Loaded] {len(test_queries)} verified Q&A test pairs from test_set_30.json.")
else:
    print("[Warning] Test dataset not found, using fallback test pairs.")
    test_queries = ["What are the prerequisites for CSE 374?"]
    ground_truths = ["CSE 374 prerequisites are CSE 143, CSE 143X, or CSE 163."]
    sample_docs = [Document(page_content="CSE 374: Intermediate Topics in Programming Techniques. Prerequisites: CSE 143, CSE 143X, or CSE 163. Covers C, C++, Linux system tools.")]

# ==========================================
# 2. Construct retriever configurations
# ==========================================

def setup_retrievers(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # A. Dense Vector Retriever
    vectorstore = Chroma.from_documents(documents, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # B. Sparse BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    return dense_retriever, bm25_retriever

# RRF (Reciprocal Rank Fusion) Hybrid Search Algorithm
def rrf_hybrid_search(query, dense_retriever, bm25_retriever, k=60, top_n=5):
    dense_docs = dense_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    
    doc_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_n]]

# Cross-Encoder Re-ranking
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_search(query, candidate_docs, top_n=3):
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]


# ==========================================
# 3. Run Ragas Evaluation Loop
# ==========================================
def evaluate_retriever(retriever_type, dense_retriever, bm25_retriever):
    retrieved_contexts_list = []
    
    for query in test_queries:
        if retriever_type == "dense":
            docs = dense_retriever.invoke(query)
        elif retriever_type == "hybrid":
            docs = rrf_hybrid_search(query, dense_retriever, bm25_retriever, top_n=5)
        elif retriever_type == "hybrid_rerank":
            candidates = rrf_hybrid_search(query, dense_retriever, bm25_retriever, top_n=10)
            docs = rerank_search(query, candidates, top_n=3)
            
        retrieved_contexts_list.append([doc.page_content for doc in docs])
        
    # Construct Dataset format required by Ragas
    dataset_dict = {
        "user_input": test_queries,
        "retrieved_contexts": retrieved_contexts_list,
        "reference": ground_truths
    }
    
    eval_dataset = Dataset.from_dict(dataset_dict)
    
    # Initialize LLM for evaluation
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Run Ragas evaluation
    results = evaluate(
        dataset=eval_dataset,
        metrics=[metric_precision, metric_recall],
        llm=evaluator_llm
    )
    
    return results

def get_score_val(results, metric_name):
    try:
        # Ragas 0.4 EvaluationResult object
        if hasattr(results, "to_pandas"):
            df = results.to_pandas()
            for col in df.columns:
                if metric_name in col:
                    val = df[col].mean()
                    if pd.notna(val):
                        return round(float(val), 3)
        # Dictionary format fallback
        if isinstance(results, dict):
            for k, v in results.items():
                if metric_name in k and isinstance(v, (int, float)):
                    return round(float(v), 3)
    except Exception as e:
        print(f"Error parsing score for {metric_name}: {e}")
    return 0.0


# ==========================================
# 4. Run Benchmark on Final Production Retriever
# ==========================================
if __name__ == "__main__":
    dense_ret, bm25_ret = setup_retrievers(sample_docs)
    
    print(f"[Testing] Running Ragas Benchmark on Final Retriever (Hybrid + Re-ranker) across {len(test_queries)} verified Q&A pairs...\n")
    
    score_rerank = evaluate_retriever("hybrid_rerank", dense_ret, bm25_ret)
    
    # Construct final production retriever results DataFrame
    df_results = pd.DataFrame([
        {
            "Retriever Architecture": "Final Production Retriever (Hybrid RAG + MiniLM Reranker)",
            "Benchmark Queries": len(test_queries),
            "Context Precision": get_score_val(score_rerank, "context_precision"),
            "Context Recall": get_score_val(score_rerank, "context_recall")
        }
    ])
    
    print("================ FINAL RETRIEVER RAGAS EVALUATION RESULTS ================")
    try:
        print(df_results.to_markdown(index=False))
    except Exception:
        print(df_results.to_string(index=False))




