"""Unit tests for Syllabus 2560 Blueprint and ExamQuestion Pydantic contracts."""

import pytest
from pydantic import ValidationError

from src.schema import (
    DatasetTier,
    DifficultyLevel,
    ExamQuestion,
    QuestionFormat,
    QuestionStyle,
    load_syllabus,
)


def _valid_mcq_payload(**overrides) -> dict:
    base = {
        "question_id": "alevel_2568_q01",
        "tier": DatasetTier.TIER_1_TARGET,
        "source_name": "A-Level Math 1 2568",
        "exam_year_be": 2568,
        "question_number": 1,
        "question_format": QuestionFormat.MCQ_5,
        "question_style": QuestionStyle.WORD_PROBLEM,
        "difficulty": DifficultyLevel.MEDIUM,
        "primary_chapter": "sequences_series",
        "secondary_chapters": ["expo_log"],
        "sub_skills": ["financial_math_compound_annuity"],
        "stem_latex": "ฝากเงินต้น $10{,}000$ บาท อัตราดอกเบี้ย $2\\%$ ต่อปี คิดดอกเบี้ยทบต้นทุกปี",
        "choices_latex": ["10200", "10404", "10612", "10824", "11040"],
        "answer": "2",
        "key_trap": "ใช้ดอกเบี้ยเชิงเดียวแทนดอกเบี้ยทบต้น",
    }
    base.update(overrides)
    return base


def test_syllabus_blueprint_bounds_are_consistent():
    syllabus = load_syllabus()
    assert syllabus.total_questions == 30
    assert len(syllabus.valid_chapter_ids()) == 16
    min_sum = sum(s.min_questions for s in syllabus.strands)
    max_sum = sum(s.max_questions for s in syllabus.strands)
    assert min_sum <= syllabus.total_questions <= max_sum


def test_valid_tier1_mcq_question_passes():
    q = ExamQuestion(**_valid_mcq_payload())
    assert q.primary_chapter == "sequences_series"
    assert q.answer == "2"


def test_valid_numeric_question_passes():
    q = ExamQuestion(
        **_valid_mcq_payload(
            question_id="alevel_2568_q26",
            question_number=26,
            question_format=QuestionFormat.NUMERIC,
            choices_latex=[],
            answer="12.5",
        )
    )
    assert q.answer == "12.5"


def test_deprecated_topic_is_rejected_hard():
    with pytest.raises(ValidationError, match="deprecated"):
        ExamQuestion(**_valid_mcq_payload(deprecated_flags=["linear_programming"]))


def test_unknown_sub_skill_is_rejected():
    with pytest.raises(ValidationError, match="Unknown sub_skills"):
        ExamQuestion(**_valid_mcq_payload(sub_skills=["made_up_skill_xyz"]))


def test_tier1_wrong_year_is_rejected():
    with pytest.raises(ValidationError, match="2564, 2569"):
        ExamQuestion(**_valid_mcq_payload(exam_year_be=2560))


def test_mcq_wrong_choice_count_is_rejected():
    with pytest.raises(ValidationError, match="exactly 5 choices"):
        ExamQuestion(**_valid_mcq_payload(choices_latex=["1", "2", "3"]))
