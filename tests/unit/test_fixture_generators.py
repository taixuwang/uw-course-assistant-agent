"""Offline checks for legacy benchmark generation and output protection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.evaluation import build_hard_benchmark, build_test_set_60


class FixtureGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.courses = [
            {"course": "E E 101", "title": "Circuit Basics", "credits": "5",
             "description": "An introduction to electrical circuits, measurements, and laboratory work."},
            {"course": "MATH 101", "title": "Mathematics", "credits": "5",
             "description": "An introduction to mathematical concepts and practical applications."},
            {"course": "OTHER 101", "title": "Other Subject", "credits": "3",
             "description": "A sufficiently long description of another field of study."},
        ]

    def test_hard_generation_is_seeded_and_recognizes_multiword_departments(self):
        first = build_hard_benchmark.build_benchmark(self.courses, seed=42)
        second = build_hard_benchmark.build_benchmark(self.courses, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first["queries"]), 2)
        self.assertIn("E E 101", first["ground_truths"][0])

    def test_basic_generation_deduplicates_course_ids(self):
        result = build_test_set_60.build_benchmark(self.courses + [self.courses[0]])
        self.assertEqual(len(result["queries"]), 3)
        self.assertEqual(len(result["queries"]), len(result["ground_truths"]))

    def test_generators_do_not_overwrite_existing_output_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "courses.json"
            output = Path(directory) / "result.json"
            catalog.write_text(json.dumps(self.courses), encoding="utf-8")
            for module in ("build_test_set_60", "build_hard_benchmark"):
                with self.subTest(module=module):
                    output.write_text("original", encoding="utf-8")
                    command = [sys.executable, "-m", f"tests.evaluation.{module}",
                               "--input", str(catalog), "--output", str(output)]
                    result = subprocess.run(command, capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
