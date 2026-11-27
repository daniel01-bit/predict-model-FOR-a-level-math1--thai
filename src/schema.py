"""Strict Pydantic contracts and Syllabus 2560 validators for Predict A-Level Math 1."""

from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_SYLLABUS_PATH = Path(__file__).resolve().parent.parent / "config" / "syllabus_2560.json"


class DatasetTier(str, Enum):
    TIER_1_TARGET = "tier_1_target"      # A-Level 66-69 + Saman IPST 64-65 (Ground Truth / Test)
    TIER_2_IPST = "tier_2_ipst"          # IPST 2560 Textbook Exercises (Universe / Gap Detector)
    TIER_3_LEGACY = "tier_3_legacy"      # Saman 55-63 + Legacy Ent (Prior / Trap Bank)


class QuestionFormat(str, Enum):
    MCQ_5 = "mcq_5"
    NUMERIC = "numeric"


class QuestionStyle(str, Enum):
    PURE_MATH = "pure_math"
    WORD_PROBLEM = "word_problem"
    VISUAL_GRAPH = "visual_graph"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ChapterSpec(BaseModel):
    id: str
    name_th: str
    sub_skills: list[str]


class StrandSpec(BaseModel):
    id: str
    name_th: str
    min_questions: int
    max_questions: int
    chapters: list[ChapterSpec]


class SyllabusConfig(BaseModel):
    curriculum_version: str
    exam_name: str
    total_questions: int
    strands: list[StrandSpec]
    deprecated_topics_hard_filter: list[str]

    def valid_chapter_ids(self) -> set[str]:
        return {ch.id for s in self.strands for ch in s.chapters}

    def valid_sub_skill_ids(self) -> set[str]:
        return {sk for s in self.strands for ch in s.chapters for sk in ch.sub_skills}

    def chapter_to_strand(self) -> dict[str, str]:
        return {ch.id: s.id for s in self.strands for ch in s.chapters}


@lru_cache(maxsize=1)
def load_syllabus(path: Path = DEFAULT_SYLLABUS_PATH) -> SyllabusConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SyllabusConfig.model_validate(data)


class ExamQuestion(BaseModel):
    """Canonical structured representation of one math question across all 3 tiers."""

    question_id: str = Field(..., description="Unique ID e.g. alevel_2568_q01")
    tier: DatasetTier
    source_name: str = Field(..., description="e.g. A-Level Math 1 2568, IPST M5 Vol2")
    exam_year_be: Optional[int] = Field(
        default=None,
        ge=2530,
        le=2575,
        description="Thai Buddhist Era year (e.g. 2568). None allowed for Tier 2 textbooks.",
    )
    question_number: Optional[int] = Field(default=None, ge=1, le=100)
    question_format: QuestionFormat
    question_style: QuestionStyle
    difficulty: DifficultyLevel

    primary_chapter: str
    secondary_chapters: list[str] = Field(default_factory=list)
    sub_skills: list[str] = Field(..., min_length=1)
    deprecated_flags: list[str] = Field(
        default_factory=list,
        description="Must be empty for accepted Syllabus 2560 items.",
    )

    stem_latex: str = Field(..., min_length=5, description="Full problem statement in Thai + LaTeX")
    choices_latex: list[str] = Field(
        default_factory=list,
        description="Exactly 5 choices for mcq_5; empty for numeric.",
    )
    answer: str = Field(..., min_length=1, description="Choice index '1'-'5' or numeric value")
    key_trap: Optional[str] = Field(
        default=None,
        description="Common distractor or conceptual trap tested by this question.",
    )
    has_diagram: bool = False

    @model_validator(mode="after")
    def validate_against_syllabus_and_format(self) -> "ExamQuestion":
        syllabus = load_syllabus()
        valid_chapters = syllabus.valid_chapter_ids()
        valid_skills = syllabus.valid_sub_skill_ids()
        deprecated_set = set(syllabus.deprecated_topics_hard_filter)

        if self.deprecated_flags:
            raise ValueError(
                f"Question {self.question_id} contains deprecated Syllabus 2560 topics: {self.deprecated_flags}"
            )

        if self.primary_chapter in deprecated_set or self.primary_chapter not in valid_chapters:
            raise ValueError(f"Invalid or deprecated primary_chapter: {self.primary_chapter!r}")

        invalid_sec = [c for c in self.secondary_chapters if c not in valid_chapters]
        if invalid_sec:
            raise ValueError(f"Invalid secondary_chapters: {invalid_sec}")

        invalid_skills = [s for s in self.sub_skills if s not in valid_skills]
        if invalid_skills:
            raise ValueError(f"Unknown sub_skills not in closed taxonomy: {invalid_skills}")

        if self.tier == DatasetTier.TIER_1_TARGET:
            if self.exam_year_be is None or not (2564 <= self.exam_year_be <= 2569):
                raise ValueError(
                    f"Tier 1 must have exam_year_be in [2564, 2569], got {self.exam_year_be}"
                )

        if self.question_format == QuestionFormat.MCQ_5:
            if len(self.choices_latex) != 5:
                raise ValueError(f"mcq_5 requires exactly 5 choices, got {len(self.choices_latex)}")
            if self.answer not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"mcq_5 answer must be '1'-'5', got {self.answer!r}")
        elif self.question_format == QuestionFormat.NUMERIC:
            if self.choices_latex:
                raise ValueError("numeric format must have empty choices_latex")
            try:
                float(self.answer)
            except ValueError as exc:
                raise ValueError(f"numeric answer must be parseable float, got {self.answer!r}") from exc

        return self
