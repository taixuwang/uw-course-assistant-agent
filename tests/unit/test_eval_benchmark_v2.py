"""Offline input and snapshot checks for the v2 benchmark runner."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.evaluation import eval_benchmark_v2
from tests.evaluation.eval_benchmark_v2 import check_reference_snapshot, load_cases


def document(code, description):
    return SimpleNamespace(metadata={"course": code},
                           page_content=f"Course: {code}\nDescription: {description}")


class BenchmarkInputTests(unittest.TestCase):
    def load(self, cases):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            return load_cases(path)

    def test_rejects_duplicate_queries_even_when_ids_are_unique(self):
        cases = [
            {"id": "f1", "category": "factual", "query": "What is CSE 101?"},
            {"id": "d1", "category": "discovery", "query": "  WHAT IS CSE 101?  "},
            {"id": "c1", "category": "constraints", "query": "Find five-credit courses"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate queries"):
            self.load(cases)

    def test_accepts_unique_queries_and_ids_for_all_categories(self):
        cases = [
            {"id": "f1", "category": "factual", "query": "What is CSE 101?"},
            {"id": "d1", "category": "discovery", "query": "Find programming courses"},
            {"id": "c1", "category": "constraints", "query": "Find five-credit courses"},
        ]
        self.assertEqual(self.load(cases)["cases"], cases)

    def test_reference_must_match_existing_index_course_description(self):
        case = {"id": "f1", "course_id": "CSE 101", "reference": "CSE 101: Course description."}
        check_reference_snapshot([case], [document("CSE 101", "Course description.")])

        with self.assertRaisesRegex(ValueError, "missing from the index"):
            check_reference_snapshot([case], [document("CSE 102", "Course description.")])
        with self.assertRaisesRegex(ValueError, "reference does not match"):
            check_reference_snapshot([case], [document("CSE 101", "Changed description.")])
        with self.assertRaisesRegex(ValueError, "reference does not match"):
            check_reference_snapshot([case], [SimpleNamespace(metadata={"course": "CSE 101"}, page_content="No description field")])

    def test_runner_saves_scores_and_ids_for_each_case_without_external_calls(self):
        doc = SimpleNamespace(metadata={"course": "CSE 101", "department": "CSE", "credits": "5"},
                              page_content="Course: CSE 101\nDescription: Course description.")
        cases = [
            {"id": "f1", "category": "factual", "query": "What is CSE 101?",
             "course_id": "CSE 101", "reference": "CSE 101: Course description."},
            {"id": "d1", "category": "discovery", "query": "Find programming courses",
             "known_course_ids": ["CSE 101"]},
            {"id": "c1", "category": "constraints", "query": "Find five-credit CSE courses",
             "constraints": {"department": "CSE", "credits": 5}},
        ]
        retriever = SimpleNamespace(documents={"CSE 101": doc}, search=lambda query: [doc])
        scores = SimpleNamespace(to_pandas=lambda: SimpleNamespace(iloc=[
            {"llm_context_precision_with_reference": 0.75, "context_recall": 1.0}
        ]))
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.json"
            output = Path(directory) / "result.json"
            fixture.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            with patch("sys.argv", ["eval_benchmark_v2", "--dataset", str(fixture), "--output", str(output)]), \
                 patch.object(eval_benchmark_v2, "create_catalog_retriever", return_value=retriever), \
                 patch.object(eval_benchmark_v2, "evaluate_data", return_value=scores) as evaluate, \
                 patch("dotenv.load_dotenv"):
                eval_benchmark_v2.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["summary"]["factual_target_hits"], 1)
            self.assertEqual(result["summary"]["discovery_known_hits"], 1)
            self.assertEqual(result["summary"]["constraint_violations_including_unknown"], 0)
            self.assertEqual([row["returned_ids"] for row in result["cases"]], [["CSE 101"]] * 3)
            self.assertEqual(evaluate.call_args.args[0]["retrieved_contexts"], [[doc.page_content]])

    def test_existing_output_is_rejected_before_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("original", encoding="utf-8")
            with patch("sys.argv", ["eval_benchmark_v2", "--output", str(output)]), \
                 patch.object(eval_benchmark_v2, "create_catalog_retriever") as create:
                with self.assertRaises(SystemExit):
                    eval_benchmark_v2.main()
            create.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
