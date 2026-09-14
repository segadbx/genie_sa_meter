"""Question validation tests: duplicate ids, missing fields, ordering, hash generation."""

from __future__ import annotations

import json

import pytest

from genie_benchmark.questions import (
    QuestionValidationError,
    load_questions,
    parse_questions,
)


def _q(**overrides):
    base = {
        "id": "q001",
        "question": "What was revenue?",
        "category": "simple_lookup",
        "expected_behavior": "Returns the metric.",
    }
    base.update(overrides)
    return base


def test_valid_questions_parse(sample_questions) -> None:
    questions = parse_questions(sample_questions)
    assert [q.id for q in questions] == ["q001", "q002"]
    assert questions[1].is_follow_up
    assert questions[1].follow_up_to == "q001"


def test_missing_field_rejected() -> None:
    bad = _q()
    del bad["category"]
    with pytest.raises(QuestionValidationError, match="category"):
        parse_questions([bad])


def test_empty_question_text_rejected() -> None:
    with pytest.raises(QuestionValidationError, match="question"):
        parse_questions([_q(question="   ")])


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(QuestionValidationError, match="Duplicate question id"):
        parse_questions([_q(id="dup"), _q(id="dup")])


def test_follow_up_to_unknown_id_rejected() -> None:
    with pytest.raises(QuestionValidationError, match="follow-up"):
        parse_questions([_q(id="q1", follow_up_to="does_not_exist")])


def test_wrapper_object_accepted() -> None:
    questions = parse_questions({"questions": [_q()]})
    assert questions[0].id == "q001"


def test_ordering_preserved() -> None:
    ids = ["q003", "q001", "q002"]
    questions = parse_questions([_q(id=i) for i in ids])
    assert [q.id for q in questions] == ids


def test_load_questions_hash_is_stable(tmp_path) -> None:
    p = tmp_path / "questions.json"
    payload = [_q()]
    p.write_text(json.dumps(payload), encoding="utf-8")
    _, hash_a = load_questions(p)
    _, hash_b = load_questions(p)
    assert hash_a == hash_b
    assert len(hash_a) == 64  # sha256 hex


def test_load_questions_does_not_modify_file(tmp_path) -> None:
    p = tmp_path / "questions.json"
    original = json.dumps([_q()], indent=4)
    p.write_text(original, encoding="utf-8")
    load_questions(p)
    assert p.read_text(encoding="utf-8") == original
