"""Benchmark question loading and input validation.

Questions are rejected if they are missing a stable ``id``, non-empty question text,
``category``, or ``expected_behavior``. Duplicate ids are rejected. The input file is
never modified; its SHA-256 hash is recorded in run metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkQuestion
from .redaction import sha256_hex

_REQUIRED_FIELDS = ("id", "question", "category", "expected_behavior")


class QuestionValidationError(ValueError):
    """Raised when the benchmark question file is invalid, with an actionable message."""


def _coerce_list(data: Any) -> list[dict[str, Any]]:
    """Accept either a top-level list or a ``{"questions": [...]}`` wrapper."""
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise QuestionValidationError(
            "Question file must be a JSON list of question objects "
            "(or an object with a 'questions' list)."
        )
    return data


def parse_questions(data: Any) -> list[BenchmarkQuestion]:
    """Validate and normalize raw question data into :class:`BenchmarkQuestion` objects."""
    raw_list = _coerce_list(data)
    if not raw_list:
        raise QuestionValidationError("Question file contains no questions.")

    seen_ids: set[str] = set()
    questions: list[BenchmarkQuestion] = []
    for index, item in enumerate(raw_list):
        if not isinstance(item, dict):
            raise QuestionValidationError(f"Question at position {index} is not an object.")
        missing = [
            f
            for f in _REQUIRED_FIELDS
            if f not in item or item[f] is None or str(item[f]).strip() == ""
        ]
        if missing:
            ident = item.get("id", f"<position {index}>")
            raise QuestionValidationError(
                f"Question '{ident}' is missing required field(s): {', '.join(missing)}."
            )
        qid = str(item["id"])
        if qid in seen_ids:
            raise QuestionValidationError(f"Duplicate question id: '{qid}'.")
        seen_ids.add(qid)
        questions.append(
            BenchmarkQuestion(
                id=qid,
                question=str(item["question"]),
                category=str(item["category"]),
                expected_behavior=str(item["expected_behavior"]),
                expected_route=_opt_str(item.get("expected_route")),
                ground_truth_notes=_opt_str(item.get("ground_truth_notes")),
                follow_up_to=_opt_str(item.get("follow_up_to")),
            )
        )

    # A follow-up question must reference a question id that exists in the set.
    for q in questions:
        if q.follow_up_to is not None and q.follow_up_to not in seen_ids:
            raise QuestionValidationError(
                f"Question '{q.id}' is a follow-up to unknown question id "
                f"'{q.follow_up_to}'."
            )
    return questions


def _opt_str(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def load_questions(path: str | Path) -> tuple[list[BenchmarkQuestion], str]:
    """Load questions from a JSON file.

    Returns the validated questions and the SHA-256 hash of the file's raw bytes. The
    file itself is read only and never modified.
    """
    p = Path(path)
    if not p.exists():
        raise QuestionValidationError(f"Question file not found: {p}")
    raw_bytes = p.read_bytes()
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise QuestionValidationError(f"Could not parse question file {p}: {exc}") from exc
    questions = parse_questions(data)
    file_hash = sha256_hex(raw_bytes.decode("utf-8"))
    return questions, file_hash
