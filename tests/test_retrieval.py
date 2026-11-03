"""Offline regression tests; no API key, browser, or model download required."""

import json
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator, StructuredQuery

from uw_course_assistant.retrieval import (
    CatalogRetriever, create_query_planner, matches_filter, validate_filter,
)


def comparison(field, value, comparator=Comparator.EQ):
    return Comparison(attribute=field, comparator=comparator, value=value)


def course(code, credits="5", prerequisite=False, gen_ed="NSc", text=None):
    return Document(page_content=text or f"Course: {code} introductory programming", metadata={
        "course": code, "department": code.rsplit(" ", 1)[0], "credits": credits,
        "has_prerequisite": prerequisite, "gen_ed": gen_ed,
    })


class Planner:
    def __init__(self, expression=None, query="introductory programming", limit=None):
        self.plan = StructuredQuery(query=query, filter=expression, limit=limit)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.plan


class VectorStore:
    def __init__(self, docs, leak=False):
        self.docs = docs
        self.calls = []
        self.leak = leak

    def get(self, include):
        return {"documents": [doc.page_content for doc in self.docs],
                "metadatas": [doc.metadata for doc in self.docs]}

    def similarity_search(self, query, k, filter):
        self.calls.append((query, k, filter))
        if self.leak:
            return self.docs[:k]
        eligible = filter["course"]["$in"]
        return [doc for doc in self.docs if doc.metadata["course"] in eligible][:k]


class Encoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return list(range(len(pairs)))


class RetrievalTests(unittest.TestCase):
    def make_retriever(self, docs, expression=None, **kwargs):
        store, planner, encoder = VectorStore(docs), Planner(expression, **kwargs), Encoder()
        return CatalogRetriever(store, planner, encoder), store, planner, encoder

    def test_both_branches_use_same_eligible_set_before_top_k(self):
        # Strong keyword distractors precede the sole eligible course by >20 docs.
        docs = [course(f"MATH {100+i}", text="introductory programming " * 5)
                for i in range(30)]
        docs += [course("CSE 121", credits="4"), course("CSE 122", prerequisite=True),
                 course("CSE 123", text="Course: CSE 123 basic computing")]
        expression = Operation(operator=Operator.AND, arguments=[
            comparison("department", "cse"), comparison("credits", 5),
            comparison("has_prerequisite", False),
        ])
        retriever, store, planner, encoder = self.make_retriever(docs, expression)
        result = retriever.search("Recommend a 5-credit CSE course without prerequisites")
        self.assertEqual([doc.metadata["course"] for doc in result], ["CSE 123"])
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(store.calls[0][2], {"course": {"$in": ["CSE 123"]}})
        self.assertEqual(len(encoder.calls[0]), 1)
        self.assertIn("CSE 123", encoder.calls[0][0][1])
        self.assertIn("without prerequisites", encoder.calls[0][0][0])

    def test_empty_eligible_set_short_circuits_both_branches(self):
        retriever, store, _, encoder = self.make_retriever([course("MATH 100")], comparison("department", "CSE"))
        class UnexpectedSparse:
            def get_batch_scores(self, *args):
                raise AssertionError("BM25 must not run for an empty eligible set")
        retriever.sparse = UnexpectedSparse()
        self.assertEqual(retriever.search("CSE courses"), [])
        self.assertEqual(store.calls, [])
        self.assertEqual(encoder.calls, [])

    def test_no_filter_searches_entire_catalog_and_honors_limit(self):
        docs = [course(f"CSE {100+i}") for i in range(8)]
        retriever, store, _, _ = self.make_retriever(docs, limit=2)
        result = retriever.search("introductory courses")
        self.assertEqual(len(result), 2)
        self.assertEqual(set(store.calls[0][2]["course"]["$in"]), {doc.metadata["course"] for doc in docs})

    def test_sparse_only_candidates_survive_and_dense_leaks_are_rejected(self):
        docs = [course("MATH 100"), course("CSE 121")]
        retriever, store, _, encoder = self.make_retriever(docs, comparison("department", "CSE"))
        store.leak = True
        result = retriever.search("programming")
        self.assertEqual([doc.metadata["course"] for doc in result], ["CSE 121"])
        self.assertEqual(len(encoder.calls[0]), 1)

    def test_course_ids_deduplicate_different_text(self):
        docs = [course("CSE 121"), course("CSE 121", text="Duplicate description")]
        retriever, _, _, encoder = self.make_retriever(docs)
        self.assertEqual(len(retriever.search("programming")), 1)
        self.assertEqual(len(encoder.calls[0]), 1)

    def test_invalid_filter_fails_before_retrieval(self):
        for expression in [comparison("unknown", "CSE"), comparison("credits", "5"),
                           comparison("department", "CSE", Comparator.GT),
                           comparison("has_prerequisite", "false")]:
            with self.subTest(expression=expression):
                retriever, store, _, encoder = self.make_retriever([course("CSE 121")], expression)
                with self.assertRaises(ValueError):
                    retriever.search("courses")
                self.assertEqual(store.calls, [])
                self.assertEqual(encoder.calls, [])

    def test_invalid_condition_is_rejected_even_with_no_matching_documents(self):
        expression = Operation(operator=Operator.AND, arguments=[
            comparison("department", "NO_MATCH"), comparison("credits", "five")])
        with self.assertRaises(ValueError):
            validate_filter(expression)

    def test_numeric_credits_and_variable_credits(self):
        expression = comparison("credits", 5)
        for value in ["5", "5, max. 15"]:
            self.assertIs(matches_filter({"credits": value}, expression), True)
        for value in ["1-5", "3/5", "*", "N/A", None]:
            self.assertIsNone(matches_filter({"credits": value}, expression))
        self.assertFalse(matches_filter({"credits": "10"}, comparison("credits", 5, Comparator.LT)))

    def test_gen_ed_matches_tokens_not_substrings(self):
        self.assertTrue(matches_filter({"gen_ed": "NSc, DIV"}, comparison("gen_ed", "nsc", Comparator.CONTAIN)))
        self.assertTrue(matches_filter({"gen_ed": "A&H/SSc"}, comparison("gen_ed", "SSc", Comparator.CONTAIN)))
        self.assertFalse(matches_filter({"gen_ed": "NSc"}, comparison("gen_ed", "C", Comparator.CONTAIN)))

    def test_nested_logic_and_unknown_metadata(self):
        expression = Operation(operator=Operator.AND, arguments=[
            comparison("department", ["CSE", "MATH"], Comparator.IN),
            Operation(operator=Operator.NOT, arguments=[comparison("has_prerequisite", True)]),
        ])
        validate_filter(expression)
        self.assertTrue(matches_filter({"department": "CSE", "has_prerequisite": False}, expression))
        self.assertIsNone(matches_filter({"department": "CSE"}, expression))
        either = Operation(operator=Operator.OR, arguments=[comparison("department", "CSE"), comparison("department", "MATH")])
        self.assertTrue(matches_filter({"department": "MATH"}, either))
        self.assertFalse(matches_filter({"department": "BIO"}, either))

    def test_query_constructor_parses_shared_filter_without_api(self):
        llm = FakeListLLM(responses=[json.dumps({
            "query": "introductory programming",
            "filter": 'and(eq("department", "CSE"), eq("credits", 5), eq("has_prerequisite", false))',
        })])
        planner = create_query_planner(llm)
        plan = planner.invoke({"query": "5-credit CSE introductory course, no prerequisites"})
        validate_filter(plan.filter)
        self.assertEqual(plan.query, "introductory programming")
        self.assertTrue(matches_filter(course("CSE 121").metadata, plan.filter))
        self.assertFalse(matches_filter(course("CSE 122", prerequisite=True).metadata, plan.filter))

    def test_query_constructor_rejects_unsupported_filter(self):
        planner = create_query_planner(FakeListLLM(responses=[
            json.dumps({"query": "courses", "filter": 'eq("unknown", "CSE")'})
        ]))
        with self.assertRaises(OutputParserException):
            planner.invoke({"query": "courses"})

    def test_real_chroma_restricts_dense_search_to_shared_ids(self):
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_core.embeddings import DeterministicFakeEmbedding

        with tempfile.TemporaryDirectory() as directory:
            store = Chroma.from_documents(
                [course("MATH 100"), course("CSE 121"), course("CSE 122", prerequisite=True)],
                DeterministicFakeEmbedding(size=8),
                persist_directory=directory,
                client_settings=Settings(anonymized_telemetry=False),
            )
            expression = Operation(operator=Operator.AND, arguments=[
                comparison("department", "CSE"), comparison("has_prerequisite", False)])
            retriever = CatalogRetriever(store, Planner(expression), Encoder())
            self.assertEqual([doc.metadata["course"] for doc in retriever.search("programming")], ["CSE 121"])
            store.delete_collection()

    def test_empty_index_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "build_vector_db"):
            self.make_retriever([])


if __name__ == "__main__":
    unittest.main()
