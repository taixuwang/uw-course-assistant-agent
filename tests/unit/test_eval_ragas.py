"""Offline tests for the evaluation adapter; no API or model downloads needed."""

from types import SimpleNamespace
import unittest

from tests.evaluation import eval_ragas


class RecordingRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        result = self.results[query]
        if isinstance(result, Exception):
            raise result
        return [SimpleNamespace(page_content=text) for text in result]


class EvaluationDataTests(unittest.TestCase):
    def test_searches_each_question_and_keeps_empty_results_in_order(self):
        retriever = RecordingRetriever({
            "first question": ["first course", "second course"],
            "no matching courses": [],
            "last question": ["third course"],
        })
        queries = ["first question", "no matching courses", "last question"]
        references = ["answer 1", "answer 2", "answer 3"]

        dataset = eval_ragas.build_evaluation_data(retriever, queries, references)

        self.assertEqual(retriever.calls, queries)
        self.assertEqual(dataset["user_input"], queries)
        self.assertEqual(dataset["reference"], references)
        self.assertEqual(dataset["retrieved_contexts"], [
            ["first course", "second course"], [], ["third course"]
        ])

    def test_mismatched_questions_and_references_fail_before_search(self):
        retriever = RecordingRetriever({})

        with self.assertRaises(ValueError):
            eval_ragas.build_evaluation_data(retriever, ["question"], [])

        self.assertEqual(retriever.calls, [])

    def test_search_failure_is_not_treated_as_no_results(self):
        error = ValueError("invalid query plan")
        retriever = RecordingRetriever({"question": error})

        with self.assertRaisesRegex(ValueError, "invalid query plan"):
            eval_ragas.build_evaluation_data(retriever, ["question"], ["answer"])

        self.assertEqual(retriever.calls, ["question"])


if __name__ == "__main__":
    unittest.main()
