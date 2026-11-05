"""Generate the legacy synthetic course-question fixture from a catalog snapshot.

This selects courses in catalog order; generated questions and references have
not been independently reviewed. It is not a representative user-query sample.
"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/courses.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests/fixtures/test_set_60.json"
TARGET_DEPARTMENTS = {
    "CSE", "MATH", "STAT", "PHYS", "BIOL", "ECON", "PSYCH", "INFO",
    "ENGL", "CHEM", "AFRAM", "ARCH", "ARCHY", "MUSIC", "HIST", "PHIL",
    "SOC", "ART", "GEOG", "LING", "POL S", "COM", "CEE", "ME", "E E", "AA",
}


def build_benchmark(courses):
    """Return legacy-format synthetic questions and course text, up to 60."""
    selected = []
    seen = set()

    def select(course):
        code = (course.get("course") or "").strip()
        title = (course.get("title") or "").strip()
        description = (course.get("description") or "").strip()
        if not code or code in seen or not title or len(description) <= 30:
            return
        seen.add(code)
        selected.append(course)

    for course in courses:
        code = (course.get("course") or "").strip()
        department = code.rsplit(" ", 1)[0] if " " in code else ""
        if department in TARGET_DEPARTMENTS:
            select(course)
            if len(selected) == 60:
                break

    if len(selected) < 60:
        for course in courses:
            select(course)
            if len(selected) == 60:
                break

    documents, queries, references = [], [], []
    for course in selected:
        code, title, description = course["course"], course["title"], course["description"]
        credits = str(course.get("credits") or "N/A")
        gen_ed = course.get("gen_ed")
        gen_ed = ", ".join(gen_ed) if isinstance(gen_ed, list) else str(gen_ed or "N/A")
        prerequisite = str(course.get("prerequisites") or "None")
        documents.append(
            f"Course: {code} - {title}\nCredits: {credits}\nGeneral Education: {gen_ed}"
            f"\nPrerequisites: {prerequisite}\nDescription: {description}"
        )
        queries.append(f"What is {code} ({title}) about and what are its prerequisites?")
        references.append(f"{code} ({title}) covers: {description} Prerequisites: {prerequisite}.")
    return {"documents": documents, "queries": queries, "ground_truths": references}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Course catalog JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON fixture")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        courses = json.load(file)
    if not isinstance(courses, list):
        parser.error("catalog must be a JSON list of courses")
    benchmark = build_benchmark(courses)
    if not benchmark["queries"]:
        parser.error("catalog contains no suitable courses")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w" if args.force else "x", encoding="utf-8") as file:
            json.dump(benchmark, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except FileExistsError:
        parser.error(f"output already exists: {args.output}; choose --output or use --force")
    print(f"Wrote {len(benchmark['queries'])} synthetic questions to {args.output}.")


if __name__ == "__main__":
    main()
