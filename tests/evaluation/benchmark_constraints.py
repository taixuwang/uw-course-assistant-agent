"""Independent metadata oracle for the constraint-only benchmark cases.

This evaluates only the explicit constraints in a fixture, not topical relevance,
and uses the supplied catalog snapshot as an imperfect source of metadata labels.
"""

import math
import re


_ALLOWED = {"department", "credits", "gen_ed", "has_prerequisite"}
_FIXED_CREDITS = re.compile(r"\d+(?:\.\d+)?(?:\s*,\s*max\.\s*\d+(?:\.\d+)?)?", re.I)


def _validate_constraints(constraints):
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be an object")
    unknown = set(constraints) - _ALLOWED
    if unknown:
        raise ValueError(f"Unsupported constraint fields: {sorted(unknown)}")
    if "department" in constraints:
        value = constraints["department"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("department must be a nonempty string")
    if "credits" in constraints:
        value = constraints["credits"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("credits must be a finite number")
    if "gen_ed" in constraints:
        value = constraints["gen_ed"]
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("gen_ed must be a nonempty list of nonempty strings")
    if "has_prerequisite" in constraints and type(constraints["has_prerequisite"]) is not bool:
        raise ValueError("has_prerequisite must be a boolean")


def _credit_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _FIXED_CREDITS.fullmatch(text):
        return None
    return float(text.split(",", 1)[0].strip())


def _gen_ed_values(value):
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            return None
        value = ",".join(value)
    if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "n/a":
        return None
    return {part.strip().casefold() for part in re.split(r"[,/]", value) if part.strip()}


def _satisfies(metadata, constraints):
    """Return True, False, or None for a match, mismatch, or unknown."""
    outcomes = []
    for field, expected in constraints.items():
        actual = metadata.get(field)
        if field == "department":
            if not isinstance(actual, str) or not actual.strip() or actual.strip().casefold() in {"unknown", "n/a"}:
                outcomes.append(None)
            else:
                outcomes.append(actual.strip().casefold() == expected.strip().casefold())
        elif field == "credits":
            actual = _credit_value(actual)
            outcomes.append(None if actual is None else actual == expected)
        elif field == "gen_ed":
            actual = _gen_ed_values(actual)
            outcomes.append(None if actual is None else all(item.strip().casefold() in actual for item in expected))
        elif field == "has_prerequisite":
            outcomes.append(None if type(actual) is not bool else actual == expected)
    if False in outcomes:
        return False
    if None in outcomes:
        return None
    return True


def evaluate_constraint_case(case, documents, returned_docs):
    """Score results against constraints and the given catalog metadata snapshot.

    Unknown metadata does not satisfy a constraint and is counted separately as
    unknown as well as a violation among returned results. A known mismatch takes
    precedence when another constraint has unknown metadata. `empty_correct` is
    only defined for cases where the snapshot contains no eligible courses.
    """
    constraints = case["constraints"]
    _validate_constraints(constraints)
    eligible_ids = set()
    seen_ids = set()
    for doc in documents:
        metadata = doc.metadata
        course = metadata.get("course")
        if not isinstance(course, str) or not course.strip():
            raise ValueError("catalog documents must have a course ID")
        if course in seen_ids:
            continue
        seen_ids.add(course)
        if _satisfies(metadata, constraints) is True:
            eligible_ids.add(course)

    returned_docs = list(returned_docs)
    outcomes = [_satisfies(doc.metadata, constraints) for doc in returned_docs]
    valid_count = sum(outcome is True for outcome in outcomes)
    unknown_count = sum(outcome is None for outcome in outcomes)
    empty_expected = not eligible_ids
    return {
        "eligible_count": len(eligible_ids),
        "returned_count": len(returned_docs),
        "valid_count": valid_count,
        "violation_count": len(returned_docs) - valid_count,
        "unknown_count": unknown_count,
        "hit": valid_count > 0,
        "empty_expected": empty_expected,
        "empty_correct": (len(returned_docs) == 0) if empty_expected else None,
    }
