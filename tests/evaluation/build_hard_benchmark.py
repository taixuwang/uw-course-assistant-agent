"""Generate the legacy synthetic hard benchmark from a catalog snapshot.

Questions use templates and can be repeated or ambiguous; references name one
course even when others qualify. They have not been independently reviewed.
Use the v2 benchmark for evaluation of broad multi-answer queries.
"""

import argparse
import json
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/courses.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests/fixtures/test_set_hard_60.json"
PRIORITY_DEPARTMENTS = {
    "CSE", "MATH", "STAT", "PHYS", "BIOL", "CHEM", "ECON", "PSYCH",
    "INFO", "ENGL", "AFRAM", "ARCH", "ARCHY", "E E", "ME", "CEE",
}


def build_benchmark(courses, seed=42):
    """Return the historical JSON shape, using a local seeded random generator."""
    valid = [course for course in courses
             if course.get("course") and course.get("title")
             and len(course.get("description") or "") > 40]

    def priority(course):
        code = course["course"].strip()
        department = code.rsplit(" ", 1)[0] if " " in code else ""
        return department in PRIORITY_DEPARTMENTS

    priority_pool = [course for course in valid if priority(course)]
    other_pool = [course for course in valid if not priority(course)]
    corpus = priority_pool[:600] + random.Random(seed).sample(other_pool, min(400, len(other_pool)))

    def format_document(course):
        gen_ed = course.get("gen_ed")
        gen_ed = ", ".join(gen_ed) if isinstance(gen_ed, list) else str(gen_ed or "N/A")
        return (
            f"Course: {course['course']} - {course['title']}\nCredits: {course.get('credits') or 'N/A'}"
            f"\nGeneral Education: {gen_ed}\nPrerequisites: {course.get('prerequisites') or 'None'}"
            f"\nDescription: {course['description']}"
        )

    queries, references = [], []
    for index, course in enumerate(priority_pool[:60]):
        code, title, description = course["course"], course["title"], course["description"]
        department = code.rsplit(" ", 1)[0] if " " in code else code
        prerequisite = str(course.get("prerequisites") or "None")
        gen_ed = course.get("gen_ed")
        gen_ed = ", ".join(gen_ed) if isinstance(gen_ed, list) else str(gen_ed or "None")
        credits = str(course.get("credits") or "N/A")
        references.append(f"Course {code} ({title}): {description} Prerequisites: {prerequisite}. Credits: {credits}.")
        category = index % 5
        if category == 0:
            short_description = description.split(".")[0] if "." in description else description[:80]
            query = f"Looking for a course covering {short_description.lower()}"
        elif category == 1:
            query = f"{department} course with prerequisites requirement: {prerequisite}"
        elif category == 2:
            query = f"{credits} credit {department} course satisfying {gen_ed} requirements"
        elif category == 3:
            words = [word for word in description.split() if len(word) > 5 and word.isalpha()]
            keywords = " ".join(words[:4]) if len(words) >= 4 else title
            query = f"Which class focuses on {keywords}?"
        else:
            query = f"Tell me about the course content for {title} in {department}"
        queries.append(query)

    return {"corpus": [format_document(course) for course in corpus],
            "queries": queries, "ground_truths": references}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Course catalog JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON fixture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for distractor sampling")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        courses = json.load(file)
    if not isinstance(courses, list):
        parser.error("catalog must be a JSON list of courses")
    benchmark = build_benchmark(courses, seed=args.seed)
    if not benchmark["queries"]:
        parser.error("catalog contains no suitable courses in the target departments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w" if args.force else "x", encoding="utf-8") as file:
            json.dump(benchmark, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except FileExistsError:
        parser.error(f"output already exists: {args.output}; choose --output or use --force")
    print(f"Wrote {len(benchmark['queries'])} synthetic questions and "
          f"{len(benchmark['corpus'])} corpus documents to {args.output} (seed {args.seed}).")


if __name__ == "__main__":
    main()
