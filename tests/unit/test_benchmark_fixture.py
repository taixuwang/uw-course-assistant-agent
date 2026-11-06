"""Structural checks for the versioned catalog benchmark fixture."""

from collections import Counter
import unittest

from tests.evaluation.benchmark_constraints import evaluate_constraint_case
from tests.evaluation.eval_benchmark_v2 import DEFAULT_BENCHMARK, load_cases


class BenchmarkFixtureTests(unittest.TestCase):
    def test_benchmark_has_one_hundred_factual_and_sixty_other_valid_cases(self):
        benchmark = load_cases(DEFAULT_BENCHMARK)
        cases = benchmark["cases"]
        self.assertEqual(Counter(case["category"] for case in cases),
                         {"factual": 100, "discovery": 30, "constraints": 30})
        factual_ids = [case["course_id"] for case in cases if case["category"] == "factual"]
        self.assertEqual(len(set(factual_ids)), 100, "Factual cases must cover distinct courses")
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["query"].strip())
                if case["category"] == "factual":
                    self.assertTrue(case["course_id"].strip())
                    self.assertTrue(case["reference"].strip())
                    self.assertIn(case["course_id"], case["query"])
                    self.assertIn(case["course_id"], case["reference"])
                elif case["category"] == "discovery":
                    self.assertTrue(case["known_course_ids"])
                    self.assertTrue(all(isinstance(code, str) and code.strip()
                                        for code in case["known_course_ids"]))
                else:
                    self.assertTrue(case["constraints"])
                    evaluate_constraint_case(case, [], [])


if __name__ == "__main__":
    unittest.main()
