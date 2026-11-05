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

    def test_retrieval_errors_are_excluded_and_checkpoint_survives_scoring_failure(self):
        doc = SimpleNamespace(metadata={"course": "CSE 101", "department": "CSE", "credits": "5"},
                              page_content="Course: CSE 101\nDescription: Course description.")
        cases = [
            {"id": "f1", "category": "factual", "query": "Course details",
             "course_id": "CSE 101", "reference": "CSE 101: Course description."},
            {"id": "d1", "category": "discovery", "query": "Find programming",
             "known_course_ids": ["CSE 101"]},
            {"id": "c1", "category": "constraints", "query": "Find five credits",
             "constraints": {"department": "CSE", "credits": 5}},
        ]
        def search(query):
            if query == "Find programming":
                raise ValueError("contain requires a text field")
            return [doc]
        retriever = SimpleNamespace(documents={"CSE 101": doc}, search=search)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.json"
            output = Path(directory) / "result.json"
            fixture.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            with patch("sys.argv", ["eval_benchmark_v2", "--dataset", str(fixture), "--output", str(output)]), \
                 patch.object(eval_benchmark_v2, "create_catalog_retriever", return_value=retriever), \
                 patch.object(eval_benchmark_v2, "evaluate_data", side_effect=RuntimeError("scoring failed")), \
                 patch("dotenv.load_dotenv"):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    eval_benchmark_v2.main()
            self.assertFalse(output.exists())
            checkpoint = json.loads(output.with_name(output.name + ".partial.json").read_text())
            self.assertEqual(len(checkpoint["cases"]), 3)
            self.assertEqual(checkpoint["cases"][1]["error_type"], "ValueError")
            self.assertNotIn("known_hit", checkpoint["cases"][1])
            self.assertEqual(checkpoint["ragas_data"]["retrieved_contexts"], [[doc.page_content]])
            scores = SimpleNamespace(to_pandas=lambda: SimpleNamespace(iloc=[
                {"llm_context_precision_with_reference": 1.0, "context_recall": 1.0}
            ]))
            with patch("sys.argv", ["eval_benchmark_v2", "--dataset", str(fixture), "--output", str(output), "--overwrite"]), \
                 patch.object(eval_benchmark_v2, "create_catalog_retriever", return_value=retriever), \
                 patch.object(eval_benchmark_v2, "evaluate_data", return_value=scores), \
                 patch("dotenv.load_dotenv"):
                eval_benchmark_v2.main()
            summary = json.loads(output.read_text())["summary"]
            self.assertEqual(summary["retrieval_failures"], 1)
            self.assertEqual(summary["discovery_cases"], 1)
            self.assertEqual(summary["discovery_evaluated"], 0)
            self.assertEqual(summary["factual_evaluated"], 1)
            self.assertEqual(summary["constraint_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
