"""Catalog retrieval: parse once, select eligible IDs once, then rank twice."""

import math
import re

from langchain_classic.chains.query_constructor.base import load_query_constructor_runnable
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator
from rank_bm25 import BM25Okapi


FIELDS = {
    "course": "string",
    "department": "string",
    "has_prerequisite": "boolean",
    "title": "string",
    "credits": "number",
    "gen_ed": "string",
    "prerequisites": "string",
}
COMPARATORS = [Comparator.EQ, Comparator.NE, Comparator.GT, Comparator.GTE,
               Comparator.LT, Comparator.LTE, Comparator.CONTAIN, Comparator.IN,
               Comparator.NIN]


def create_query_planner(llm):
    return load_query_constructor_runnable(
        llm,
        "University of Washington courses. Extract every explicit hard constraint "
        "into the filter and keep topic/skill preferences in the search query. "
        "Do not invent hard constraints for subjective preferences like introductory. "
        'For general education, quote the field name and value, for example '
        'contain("gen_ed", "SSc").',
        [AttributeInfo(name=name, type=kind, description=description)
         for name, kind, description in [
             ("course", "string", "Exact course code, e.g. CSE 374"),
             ("department", "string", "Full department code, e.g. CSE or CHID"),
             ("has_prerequisite", "boolean", "Whether prerequisites are listed"),
             ("title", "string", "Course title"),
             ("credits", "number", "Fixed credits per enrollment, e.g. 5. Variable or unknown credits do not match numeric filters."),
             ("gen_ed", "string", "General education codes: A&H, SSc, NSc, DIV, RSN, C, W. Use contain for membership."),
             ("prerequisites", "string", "Prerequisites text"),
         ]],
        allowed_comparators=COMPARATORS,
        allowed_operators=[Operator.AND, Operator.OR, Operator.NOT],
        # The result count is applied after fusion/reranking, never to one branch.
        enable_limit=True,
        fix_invalid=False,
    )


def _value(attribute, value):
    kind = FIELDS[attribute]
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{attribute} requires a boolean")
        return value
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{attribute} requires a finite number")
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{attribute} requires a nonempty string")
    return " ".join(value.split()).casefold()


def validate_filter(expression):
    """Reject unsupported conditions before touching either retrieval branch."""
    if expression is None:
        return
    if isinstance(expression, Operation):
        if expression.operator not in (Operator.AND, Operator.OR, Operator.NOT):
            raise ValueError("Unsupported logical operator")
        if not expression.arguments or (expression.operator == Operator.NOT and len(expression.arguments) != 1):
            raise ValueError("Invalid logical filter arguments")
        for argument in expression.arguments:
            validate_filter(argument)
        return
    if not isinstance(expression, Comparison) or expression.attribute not in FIELDS:
        raise ValueError("Unsupported filter field")
    comparator, attribute = expression.comparator, expression.attribute
    if comparator not in COMPARATORS:
        raise ValueError("Unsupported comparator")
    if comparator in (Comparator.GT, Comparator.GTE, Comparator.LT, Comparator.LTE) and attribute != "credits":
        raise ValueError("Ordered comparisons require numeric credits")
    if comparator == Comparator.CONTAIN and FIELDS[attribute] != "string":
        raise ValueError("contain requires a text field")
    values = expression.value
    if comparator in (Comparator.IN, Comparator.NIN):
        if not isinstance(values, list) or not values:
            raise ValueError("in/nin requires a nonempty list")
    else:
        values = [values]
    for value in values:
        _value(attribute, value)


def _metadata_value(metadata, attribute):
    value = metadata.get(attribute)
    if attribute == "credits":
        # '5, max. 15' is fixed at 5 per enrollment; '1-5' and '3/5' are variable.
        fixed = str(value).split(",", 1)[0].strip()
        return float(fixed) if re.fullmatch(r"\d+(?:\.\d+)?", fixed) else None
    if attribute == "gen_ed":
        if isinstance(value, list):
            value = ",".join(value)
        if not isinstance(value, str) or value in ("", "N/A"):
            return None
        return {part.strip().casefold() for part in re.split(r"[,/]", value)}
    try:
        return _value(attribute, value)
    except ValueError:
        return None


def matches_filter(metadata, expression):
    """Three-valued evaluation: missing metadata never satisfies NOT by accident."""
    if expression is None:
        return True
    if isinstance(expression, Operation):
        values = [matches_filter(metadata, arg) for arg in expression.arguments]
        if expression.operator == Operator.NOT:
            return None if values[0] is None else not values[0]
        if expression.operator == Operator.AND:
            return False if False in values else (None if None in values else True)
        return True if True in values else (None if None in values else False)
    attribute, comparator = expression.attribute, expression.comparator
    actual = _metadata_value(metadata, attribute)
    if actual is None:
        return None
    if comparator in (Comparator.IN, Comparator.NIN):
        expected = [_value(attribute, value) for value in expression.value]
        result = bool(actual.intersection(expected)) if isinstance(actual, set) else actual in expected
        return result if comparator == Comparator.IN else not result
    expected = _value(attribute, expression.value)
    if comparator in (Comparator.EQ, Comparator.NE):
        result = expected in actual if isinstance(actual, set) else actual == expected
        return result if comparator == Comparator.EQ else not result
    if comparator == Comparator.CONTAIN:
        return expected in actual
    if comparator == Comparator.GT:
        return actual > expected
    if comparator == Comparator.GTE:
        return actual >= expected
    if comparator == Comparator.LT:
        return actual < expected
    return actual <= expected


def tokenize(text):
    return re.findall(r"\w+", text.casefold()) or [""]


class CatalogRetriever:
    def __init__(self, vectorstore, query_planner, cross_encoder):
        self.vectorstore = vectorstore
        self.query_planner = query_planner
        self.cross_encoder = cross_encoder
        # Use the persisted catalog as the common source for both branches.
        snapshot = vectorstore.get(include=["documents", "metadatas"])
        self.documents = {}
        for text, metadata in zip(snapshot["documents"], snapshot["metadatas"]):
            if not metadata or not metadata.get("course") or not text:
                raise ValueError("Catalog index contains a course without text or course ID")
            self.documents.setdefault(metadata["course"], Document(page_content=text, metadata=metadata))
        if not self.documents:
            raise ValueError("Catalog index is empty. Run scripts/build_vector_db.py first.")
        self.course_ids = list(self.documents)
        self.sparse = BM25Okapi([tokenize(doc.page_content) for doc in self.documents.values()])

    def search(self, question):
        plan = self.query_planner.invoke({"query": question})
        validate_filter(plan.filter)
        if plan.limit is not None and (type(plan.limit) is not int or plan.limit < 1):
            raise ValueError("Result limit must be a positive integer")
        # Select S once, before either branch performs retrieval or truncation.
        eligible = {code for code, doc in self.documents.items()
                    if matches_filter(doc.metadata, plan.filter) is True}
        if not eligible:
            return []
        query = plan.query.strip()
        if not query:
            query = question
        dense = self.vectorstore.similarity_search(
            query, k=min(20, len(eligible)),
            filter={"course": {"$in": sorted(eligible)}},
        )
        # Global IDF statistics are reused, but only members of S are scored/ranked.
        indices = [i for i, code in enumerate(self.course_ids) if code in eligible]
        scores = self.sparse.get_batch_scores(tokenize(query), indices)
        sparse = [self.documents[self.course_ids[index]] for index, _ in
                  sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)[:20]]
        fusion = {}
        for branch in (dense, sparse):
            seen = set()
            for rank, doc in enumerate(branch, start=1):
                code = doc.metadata.get("course")
                if code in eligible and code not in seen:
                    fusion[code] = fusion.get(code, 0) + 1 / (60 + rank)
                    seen.add(code)
        candidates = [self.documents[code] for code in
                      sorted(fusion, key=fusion.get, reverse=True)[:20]]
        if not candidates:
            return []
        scores = self.cross_encoder.predict([[question, doc.page_content] for doc in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        limit = min(5, plan.limit) if plan.limit is not None else 5
        return [doc for doc, _ in ranked
                if doc.metadata["course"] in eligible][:limit]
