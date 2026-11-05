"""Offline checks for the independent benchmark constraint oracle."""

import unittest

from langchain_core.documents import Document

from tests.evaluation.benchmark_constraints import evaluate_constraint_case


def course(code, *, department="CSE", credits="5", gen_ed="SSc", prereq=False):
    metadata = {"course": code, "department": department, "credits": credits,
                "gen_ed": gen_ed}
    if prereq is not None:
        metadata["has_prerequisite"] = prereq
    return Document(page_content=code, metadata=metadata)


class ConstraintBenchmarkTests(unittest.TestCase):
    def test_multiple_eligible_and_known_mismatch_and_unknown(self):
        first = course("CSE 101")
        second = course("CSE 102")
        wrong_department = course("MATH 101", department="MATH")
        wrong_credits = course("CSE 103", credits="4")
        unknown_prerequisite = course("CSE 104", prereq=None)
        case = {"constraints": {"department": "CSE", "credits": 5,
                                "has_prerequisite": False}}

        result = evaluate_constraint_case(
            case, [first, second, wrong_department, wrong_credits, unknown_prerequisite],
            [second, wrong_department, wrong_credits, unknown_prerequisite],
        )

        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["returned_count"], 4)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["violation_count"], 3)
        self.assertEqual(result["unknown_count"], 1)
        self.assertTrue(result["hit"])
        self.assertFalse(result["empty_expected"])
        self.assertIsNone(result["empty_correct"])

    def test_no_eligible_courses_correct_empty_and_incorrect_nonempty(self):
        doc = course("CSE 101")
        case = {"constraints": {"department": "MATH"}}
        empty = evaluate_constraint_case(case, [doc], [])
        nonempty = evaluate_constraint_case(case, [doc], [doc])

        self.assertTrue(empty["empty_expected"])
        self.assertTrue(empty["empty_correct"])
        self.assertFalse(empty["hit"])
        self.assertEqual(empty["eligible_count"], 0)
        self.assertTrue(nonempty["empty_expected"])
        self.assertFalse(nonempty["empty_correct"])
        self.assertEqual(nonempty["violation_count"], 1)

    def test_split_gen_ed_and_fixed_vs_variable_credit(self):
        fixed = course("CSE 101", credits="5, max. 15", gen_ed="A&H/SSc, DIV")
        variable = course("CSE 102", credits="1-5", gen_ed="A&H/SSc, DIV")
        wrong_gen_ed = course("CSE 103", credits="5", gen_ed="NSc, DIV")
        case = {"constraints": {"credits": 5, "gen_ed": ["SSc", "DIV"]}}

        result = evaluate_constraint_case(case, [fixed, variable, wrong_gen_ed],
                                          [fixed, variable, wrong_gen_ed])

        self.assertEqual(result["eligible_count"], 1)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["violation_count"], 2)
        self.assertEqual(result["unknown_count"], 1)

    def test_unknown_does_not_mask_known_mismatch(self):
        doc = course("MATH 101", department="MATH", prereq=None)
        case = {"constraints": {"department": "CSE", "has_prerequisite": False}}
        result = evaluate_constraint_case(case, [doc], [doc])
        self.assertEqual(result["violation_count"], 1)
        self.assertEqual(result["unknown_count"], 0)


if __name__ == "__main__":
    unittest.main()
